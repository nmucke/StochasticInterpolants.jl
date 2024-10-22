using Lux
using StochasticInterpolants
using Random
using CUDA
using NNlib

function MultipleBlocks(
    in_channels::Int, 
    out_channels::Int; 
    num_blocks::Int = 2, 
    multiplier::Int = 1,
    padding="periodic"
)
    channels = [in_channels, out_channels]
    for _ in 1:num_blocks-1
        push!(channels, out_channels)
    end

    return @compact(
        blocks = [conv_next_block_no_pars(
            in_channels=channels[i], 
            out_channels=channels[i+1], 
            multiplier=multiplier, 
            padding=padding
        ) for i = 1:num_blocks]
    ) do x
        for block = blocks
            x = block(x)
        end
        @return x
    end
end

function DownBlock(
    in_channels::Int, 
    out_channels::Int; 
    multiplier::Int = 1,
    num_blocks=2,
    padding="periodic",
)
    return @compact(
        conv_blocks = MultipleBlocks(in_channels, out_channels; num_blocks=num_blocks, multiplier=multiplier, padding=padding),
        down_conv = Conv((4, 4), out_channels => out_channels, stride=2, pad=1, use_bias=true),
    ) do x
        x = conv_blocks(x)
        @return down_conv(x)
    end 
end

function UpBlock(
    in_channels::Int, 
    out_channels::Int; 
    multiplier::Int = 1,
    num_blocks=2,
    padding="periodic",
)
    return @compact(
        up_conv = ConvTranspose((4, 4), in_channels => in_channels, stride=2, pad=1, use_bias=true),
        conv_blocks = MultipleBlocks(in_channels, out_channels; num_blocks=num_blocks, multiplier=multiplier, padding=padding),
    ) do x
        x = up_conv(x)
        @return conv_blocks(x)
    end
end

function Encoder(
    in_channels::Int = 3,
    channels=[32, 64, 128], 
    padding="periodic"
)
    return @compact(
        init_block = MultipleBlocks(in_channels, channels[1]; num_blocks=1, multiplier=2, padding=padding),
        down_blocks = Chain([DownBlock(channels[i], channels[i+1]; multiplier=2, padding=padding) for i = 1:length(channels)-1])
    ) do x
        x = init_block(x)
        @return down_blocks(x)
    end
end

function VariationalEncoder(
    in_channels::Int = 3,
    num_latent_channels=128,
    channels=[32, 64, 128], 
    padding="periodic",
)
    return @compact(
        init_block = MultipleBlocks(in_channels, channels[1]; num_blocks=2, multiplier=2, padding=padding),
        down_blocks = Chain([DownBlock(channels[i], channels[i+1]; multiplier=2, num_blocks=1, padding=padding) for i = 1:length(channels)-1]),
        final_block = MultipleBlocks(channels[end], channels[end]; num_blocks=1, multiplier=2, padding=padding),
        mean_conv = Conv((3, 3), channels[end] => num_latent_channels, stride=1, pad=1, use_bias=false),
        var_conv = Conv((3, 3), channels[end] => num_latent_channels, stride=1, pad=1, use_bias=false)
    ) do x
        x = init_block(x)
        x = down_blocks(x)
        x = final_block(x)

        x_mean = mean_conv(x)
        x_log_var = var_conv(x)

        @return (x_mean, x_log_var)
    end
end

function Decoder(
    out_channels::Int = 128,
    num_latent_channels=128,
    channels=[128, 64, 32], 
    padding="periodic"
)
    return @compact(
        init_block = MultipleBlocks(num_latent_channels, channels[1]; num_blocks=1, multiplier=2, padding=padding),
        up_blocks = Chain([UpBlock(channels[i], channels[i+1]; multiplier=2, num_blocks=1, padding=padding) for i = 1:length(channels)-1]),
        final_block = MultipleBlocks(channels[end], out_channels; num_blocks=1, multiplier=2, padding=padding),
        final_conv = Conv((3, 3), (out_channels => out_channels); pad=1, use_bias=false)
    ) do x
        x = init_block(x)
        x = up_blocks(x)
        x = final_block(x)
        @return final_conv(x)
    end
end

"""
    Autoencoder(in_channels::Int = 3, channels=[32, 64, 128], padding="periodic")

Create an autoencoder model with the given number of input channels, output channels, and padding type.
"""
struct Autoencoder <: Lux.AbstractExplicitContainerLayer{
    (:encoder, :decoder)
}
    encoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
end

function Autoencoder(
    in_channels::Int = 3,
    channels=[32, 64, 128], 
    padding="periodic"
)
    return Autoencoder(
        Encoder(in_channels, channels, padding),
        Decoder(in_channels, reverse(channels), padding)
    )
end

function (AE::Autoencoder)(
    x, ps::NamedTuple, st::NamedTuple
)
    x, st_new = AE.encoder(x, ps.encoder, st.encoder)
    @set! st.encoder = st_new

    x, st_new = AE.decoder(x, ps.decoder, st.decoder)
    @set! st.decoder = st_new

    return x, st    
end

struct VariationalAutoencoder <: Lux.AbstractExplicitContainerLayer{
    (:encoder, :decoder)
}
    encoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
    encode::Function
    decode::Function
    latent_dimensions::Tuple
    image_size::Tuple
end

function VariationalAutoencoder(
    in_channels::Int=3,
    image_size::Tuple=(128, 128),
    num_latent_channels=128,
    channels=[32, 64, 128], 
    padding="periodic"
)
    encoder = VariationalEncoder(in_channels,num_latent_channels, channels, padding)
    decoder = Decoder(in_channels, num_latent_channels, reverse(channels), padding)

    encode(x, ps, st) = begin
        x, st_new = encoder(x, ps.encoder, st.encoder)
        @set! st.encoder = st_new
        return x, st
    end

    decode(x, ps, st) = begin
        x, st_new = decoder(x, ps.decoder, st.decoder)
        @set! st.decoder = st_new
        return x, st
    end

    (H, W) = image_size
    dim_reduction = 2^(length(channels)-1)
    latent_dimensions = Int.((H/dim_reduction, W/dim_reduction, num_latent_channels))

    return VariationalAutoencoder(
        encoder, decoder, encode, decode, latent_dimensions, image_size
    )
end

struct VAE_wrapper
    encode::Function
    decode::Function
    latent_dimensions::Tuple
end

function VAE_wrapper(
    autoencoder, ps::NamedTuple, st::NamedTuple
)
    encode(x) = autoencoder.encode(x, ps, st)[1]
    decode(x) = autoencoder.decode(x, ps, st)[1]

    return VAE_wrapper(encode, decode, autoencoder.latent_dimensions)
end


# function (VAE::VariationalAutoencoder)(
#     x, ps::NamedTuple, st::NamedTuple
# )
#     (x_mean, x_std), st_new = VAE.encoder(x, ps.encoder, st.encoder)
#     @set! st.encoder = st_new

#     x, st_new = VAE.decoder(x_mean, ps.decoder, st.decoder)
#     @set! st.decoder = st_new

#     return x, st    
# end

