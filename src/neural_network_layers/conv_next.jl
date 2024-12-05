using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding
using LinearAlgebra

using Lux.Experimental: @compact 
using Lux.Experimental: @kwdef
using Lux.Experimental: @concrete


###############################################################################
# Conv Next Block
###############################################################################


"""
    conv_next_block_no_pars(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        pars_embed_dim::Int = 1,
        imsize::Tuple{Int, Int} = (64, 128)
    )

Create a conv next block with the given number of input and output channels. 
The block consists of two convolutional layers with kernel size `kernel_size`.
The first layer has the same number of input and output channels, while the 
second layer has the same number of output channels as the block. 
The block also includes batch normalization and a skip connection.

https://arxiv.org/abs/2201.03545

Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/b9b33913b99ede88d9452c5ab470c5d7f5da5c56/src/turbpred/model_diffusion_blocks.py#L60

"""
function conv_next_block_no_pars(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    padding="constant"
)

    @compact(
        ds_conv = Chain(
            get_padding(padding, 3),
            Lux.Conv((7, 7), in_channels => in_channels)#, groups=in_channels)
        ),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels => in_channels * multiplier)),
            NNlib.gelu,
            Lux.InstanceNorm(in_channels * multiplier),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels * multiplier => out_channels)),
        ),
        res_conv = Lux.Conv((1, 1), (in_channels => out_channels); pad=0)
    ) do x
        h = ds_conv(x)
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

"""
    multiple_conv_next_blocks_no_pars(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        padding="constant"
    )

Create a chain of two conv next blocks with the given number of input and
output channels. The first block has the same number of input and output

"""
function multiple_conv_next_blocks_no_pars(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    padding="constant"
)
    @compact(
        block_1 = conv_next_block_no_pars(
            in_channels=in_channels, 
            out_channels=out_channels, 
            multiplier=multiplier,
            padding=padding
        ),
        block_2 = conv_next_block_no_pars(
            in_channels=out_channels, 
            out_channels=out_channels, 
            multiplier=multiplier,
            padding=padding
        )
    ) do x
        x = block_1(x)
        @return block_2()
    end
end



"""
    conv_next_block(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        pars_embed_dim::Int = 1,
        imsize::Tuple{Int, Int} = (64, 128)
    )

Create a conv next block with the given number of input and output channels. 
The block consists of two convolutional layers with kernel size `kernel_size`.
The first layer has the same number of input and output channels, while the 
second layer has the same number of output channels as the block. 
The block also includes batch normalization and a skip connection.

https://arxiv.org/abs/2201.03545

Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/b9b33913b99ede88d9452c5ab470c5d7f5da5c56/src/turbpred/model_diffusion_blocks.py#L60

"""
function conv_next_block(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    pars_embed_dim::Int = 1,
    imsize::Tuple{Int, Int} = (64, 128),
    padding="constant"
)

    @compact(
        ds_conv = Chain(
            get_padding(padding, 3),
            Lux.Conv((7, 7), in_channels => in_channels)#, groups=in_channels)
        ),
        pars_mlp = Chain(
            Lux.Dense(pars_embed_dim => in_channels)
        ),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels => in_channels * multiplier)),
            NNlib.gelu,
            Lux.InstanceNorm(in_channels * multiplier),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels * multiplier => out_channels)),
        ),
        res_conv = Lux.Conv((1, 1), (in_channels => out_channels); pad=0)
    ) do x
        x, pars = x
        h = ds_conv(x)
        pars = pars_mlp(pars)
        pars = reshape(pars, 1, 1, size(pars)...)
        h = h .+ pars
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

"""
    multiple_conv_next_blocks(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        embedding_dims::Int = 1,
        imsize::Tuple{Int, Int} = (64, 128)
    )

Create a chain of two conv next blocks with the given number of input and
output channels. The first block has the same number of input and output
"""
function multiple_conv_next_blocks(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dims::Int = 1,
    imsize::Tuple{Int, Int} = (64, 128),
    padding="constant"
)
    @compact(
        block_1 = conv_next_block(
            in_channels=in_channels, 
            out_channels=out_channels, 
            multiplier=multiplier,
            pars_embed_dim=embedding_dims,
            imsize=imsize,
            padding=padding
        ),
        block_2 = conv_next_block(
            in_channels=out_channels, 
            out_channels=out_channels, 
            multiplier=multiplier,
            pars_embed_dim=embedding_dims,
            imsize=imsize,
            padding=padding
        )
    ) do x
        x, t = x
        # @return block_1((x, t))
        x = block_1((x, t))
        @return block_2((x, t))
    end
end

###############################################################################
# Conv Next UNet without Pars
###############################################################################

function ConvNextUNetNoPars(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    attention_type="linear",
    use_attention_in_layer=[false, false, true, true],
    attention_embedding_dims=32,
    num_heads=4,
    padding="periodic"
)


    conv_down_blocks = []
    attn_down_blocks = []
    down_blocks = []
    return @compact(

        )
    ) do x
    end


###############################################################################
# Attn Pars Conv Next UNet
###############################################################################

"""
AttnParsConvNextUNet(
        image_size::Tuple{Int, Int},
        channels::Vector{Int} = [32, 64, 96, 128],
        block_depth::Int = 2,
        min_freq::Float32 = 1.0f0,
        max_freq::Float32 = 1000.0f0,
        embedding_dims::Int = 32
    )

Create a conditional U-Net model with the given image size, number of channels,
block depth, frequency range, and embedding dimensions.
"""
struct AttnParsConvNextUNet <: Lux.AbstractExplicitContainerLayer{
    (
        :conv_in, :conv_out, :conv_down_blocks, 
        :down_blocks, :bottleneck1, :bottleneck2, :conv_up_blocks, 
        :up_blocks, :t_embedding, :pars_embedding, :attn_down_blocks,
        :bottleneck_attn, :attn_up_blocks
    )
}
    # layes::LayersType
    conv_in::Lux.AbstractExplicitLayer
    conv_out::Lux.AbstractExplicitLayer
    conv_down_blocks::Lux.AbstractExplicitLayer
    down_blocks::Lux.AbstractExplicitLayer
    bottleneck1::Lux.AbstractExplicitLayer
    bottleneck2::Lux.AbstractExplicitLayer
    conv_up_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    t_embedding::Lux.AbstractExplicitLayer
    pars_embedding::Lux.AbstractExplicitLayer
    attn_down_blocks::Lux.AbstractExplicitLayer
    bottleneck_attn::Lux.AbstractExplicitLayer
    attn_up_blocks::Lux.AbstractExplicitLayer
    len_history::Int
end

function AttnParsConvNextUNet(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32,
    pars_dim=32,
    len_history=1,
    attention_type="linear",
    use_attention_in_layer=[false, false, true, true],
    attention_embedding_dims=32,
    num_heads=4,
    padding="periodic"
)


    attention_layer(imsize, in_channels, embed_dim, num_heads) = get_attention_layer(
        attention_type, imsize, in_channels, embed_dim, num_heads, embedding_dims
    )
    
    multiplier = 2

    if pars_dim == 0
        t_pars_embedding_dims = embedding_dims

        pars_embedding = Chain(
            a -> transform_to_nothing(a),
        )
    else
        t_pars_embedding_dims = div(embedding_dims, 2)

        pars_embedding = Chain(
            x -> sinusoidal_embedding(x, min_freq, max_freq, t_pars_embedding_dims),
            Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
            NNlib.gelu,
            Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
            NNlib.gelu,
        )

    end

    init_channels = channels[1]
    # init_channels = div(channels[1], 2)

    conv_in = conv_next_block_no_pars(
        in_channels=len_history*in_channels + in_channels, 
        out_channels=init_channels, 
        multiplier=multiplier,
        padding=padding
    )   

    t_embedding = Chain(
        x -> sinusoidal_embedding(x, min_freq, max_freq, t_pars_embedding_dims),
        Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
        NNlib.gelu,
        Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
        NNlib.gelu,
    )

    conv_down_blocks = []
    attn_down_blocks = []
    down_blocks = []
    for i in 1:(length(channels) - 1)
        imsize = div.(image_size, 2^(i - 1))

        push!(
            conv_down_blocks, 
            multiple_conv_next_blocks(
                in_channels=channels[i], 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                embedding_dims=embedding_dims,
                imsize=imsize,
                padding=padding
            )
        )

        if use_attention_in_layer[i]
            push!(attn_down_blocks, attention_layer(
                imsize, channels[i + 1], attention_embedding_dims, num_heads,
            ))
        else
            push!(attn_down_blocks, StateParsIdentity())
        end

        push!(down_blocks, Conv(
            (4, 4), 
            (channels[i + 1] => channels[i + 1]); 
            use_bias=true,
            pad=1,
            stride=2
        ))
    end

    conv_down_blocks = Chain(conv_down_blocks...)#; disable_optimizations=true)
    attn_down_blocks = Chain(attn_down_blocks...)#; disable_optimizations=true)
    down_blocks = Chain(down_blocks...)#; disable_optimizations=true)

    imsize = div.(image_size, 2^(length(channels) - 1))
    
    bottleneck1 = conv_next_block(
        in_channels=channels[end], 
        out_channels=channels[end],
        multiplier=multiplier,
        pars_embed_dim=embedding_dims,
        imsize=imsize,
        padding=padding
    )


    bottleneck_attn = attention_layer(
        imsize, channels[end], attention_embedding_dims, num_heads
    )

    bottleneck2 = conv_next_block(
        in_channels=channels[end], 
        out_channels=channels[end],
        multiplier=multiplier,
        pars_embed_dim=embedding_dims,
        imsize=imsize,
        padding=padding
    )

    reverse!(channels)
    reverse!(use_attention_in_layer)
    conv_up_blocks = []
    attn_up_blocks = []
    up_blocks = []
    for i in 1:(length(channels) - 1)
        push!(up_blocks, ConvTranspose(
            (4, 4), 
            (channels[i] => channels[i]); 
            use_bias=true,
            pad=1,
            stride=2
        ))
        
        imsize = div.(image_size, 2^(length(channels) - i - 1))
        push!(
            conv_up_blocks, 
            multiple_conv_next_blocks(
                in_channels=channels[i] * 2, 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                embedding_dims=embedding_dims,
                imsize=imsize,
                padding=padding
            )
        )

        if use_attention_in_layer[i]
            push!(attn_up_blocks, attention_layer(
                imsize, channels[i + 1], attention_embedding_dims, num_heads
            ))
        else
            push!(attn_up_blocks, StateParsIdentity())
        end
    end

    conv_up_blocks = Chain(conv_up_blocks...)#; disable_optimizations=true)
    attn_up_blocks = Chain(attn_up_blocks...)#; disable_optimizations=true)
    up_blocks = Chain(up_blocks...)#; disable_optimizations=true)

    conv_out = Chain(
        conv_next_block_no_pars(
            in_channels=channels[end], 
            out_channels=channels[end], 
            multiplier=multiplier,
            padding=padding
        ),
        Conv((1, 1), (channels[end] => in_channels); use_bias=false)
    )
    
    return AttnParsConvNextUNet(
        conv_in, conv_out, conv_down_blocks, 
        down_blocks, bottleneck1, bottleneck2, conv_up_blocks, 
        up_blocks, t_embedding, pars_embedding, attn_down_blocks,
        bottleneck_attn, attn_up_blocks, len_history
    )
end


function (conv_next_unet::AttnParsConvNextUNet)(
    x,#::Tuple{AbstractArray{T, 4}, AbstractArray{T, 4}, AbstractArray{S, 4}}, 
    ps::NamedTuple,
    st::NamedTuple
)# where T <: AbstractFloat

    x, x_0, pars, t = x

    H, W, C, len_history, B = size(x_0)
    x_0 = reshape(x_0, H, W, C*len_history, B);

    t_emb, new_st = conv_next_unet.t_embedding(t, ps.t_embedding, st.t_embedding)
    @set! st.t_embedding = new_st

    pars, new_st = conv_next_unet.pars_embedding(pars, ps.pars_embedding, st.pars_embedding)
    @set! st.pars_embedding = new_st

    t_emb = pars_cat(t_emb, pars; dims=1)

    x = cat(x, x_0; dims=3)

    x, new_st = conv_next_unet.conv_in(x, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    
    skips = (x, )
    for i in 1:length(conv_next_unet.conv_down_blocks)

        conv_layer_name = Symbol(:layer_, i)
        x, new_st = conv_next_unet.conv_down_blocks[i](
            (x, t_emb), ps.conv_down_blocks[conv_layer_name], st.conv_down_blocks[conv_layer_name]
        )  
        @set! st.conv_down_blocks[conv_layer_name] = new_st

        x, new_st = conv_next_unet.attn_down_blocks[i](
            (x, t_emb), ps.attn_down_blocks[conv_layer_name], st.attn_down_blocks[conv_layer_name]
        )
        @set! st.attn_down_blocks[conv_layer_name] = new_st

        skips = (skips..., x)

        x, new_st = conv_next_unet.down_blocks[i](
            x, ps.down_blocks[conv_layer_name], st.down_blocks[conv_layer_name]
        )
        @set! st.down_blocks[conv_layer_name] = new_st

    end

    x, new_st = conv_next_unet.bottleneck1(
        (x, t_emb), ps.bottleneck1, st.bottleneck1
    )
    @set! st.bottleneck1 = new_st

    x, new_st = conv_next_unet.bottleneck_attn(
        (x, t_emb), ps.bottleneck_attn, st.bottleneck_attn
    )
    @set! st.bottleneck_attn = new_st

    x, new_st = conv_next_unet.bottleneck2(
        (x, t_emb), ps.bottleneck2, st.bottleneck2
    )
    @set! st.bottleneck2 = new_st


    for i in 1:length(conv_next_unet.conv_up_blocks)
        layer_name = Symbol(:layer_, i)

        x, new_st = conv_next_unet.up_blocks[i](
            x, ps.up_blocks[layer_name], st.up_blocks[layer_name]
        )
        @set! st.up_blocks[layer_name] = new_st

        x = cat(x, skips[end-i+1]; dims=3) # cat on channel  
        # x = x + skips[end-i+1]
              
        x, new_st = conv_next_unet.conv_up_blocks[i](
            (x, t_emb), ps.conv_up_blocks[layer_name], st.conv_up_blocks[layer_name]
        )
        @set! st.conv_up_blocks[layer_name] = new_st
        
        x, new_st = conv_next_unet.attn_up_blocks[i](
            (x, t_emb), ps.attn_up_blocks[layer_name], st.attn_up_blocks[layer_name]
        )
        @set! st.attn_up_blocks[layer_name] = new_st
        
    end

    x, new_st = conv_next_unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end


