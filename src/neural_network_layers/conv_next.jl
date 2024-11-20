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

function pars_cat(x::AbstractArray, y::Nothing; dims=1)
    return x
end

function pars_cat(x::AbstractArray, y::AbstractArray; dims=1)
    return cat(x, y; dims=dims)
end


function transform_to_nothing(x::AbstractArray)
    return nothing
end

"""
    ViPosEmbedding(embedding_size, number_patches; init = randn32)

Positional embedding layer used by many vision transformer-like models.
"""
@kwdef @concrete struct ViPosEmbedding <: Lux.AbstractExplicitLayer
    embedding_size::Int
    number_patches::Int
    init = randn32
end

@inline ViPosEmbedding(embedding_size::Int, number_patches::Int; init=randn32) = ViPosEmbedding(
    embedding_size, number_patches, init)

function LuxCore.initialparameters(rng::AbstractRNG, v::ViPosEmbedding)
    return (; vectors=v.init(rng, v.embedding_size, v.number_patches))
end

@inline (v::ViPosEmbedding)(x, ps, st) = x .+ ps.vectors, st

function Identity()
    @compact(
        identity = nothing
    ) do x
        x = x
        @return x
    end
end



function StateParsIdentity()
    @compact(
        identity = nothing
    ) do x
        x, pars = x
        @return x
    end
end

function get_padding(padding::String, padding_size::Int)

    if padding == "constant"
        return a -> pad(a, 0, (padding_size, padding_size))
    elseif padding == "smooth"
        return a -> pad(a, :smooth, (padding_size, padding_size))
    elseif padding == "periodic"
        return a -> pad(a, :periodic, (padding_size, padding_size))
    else
        return a -> a
    end
end



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



function get_attention_layer(
    attention_type,
    imsize,
    in_channels,
    embed_dim,
    num_heads,
    t_pars_embedding_dims=nothing,
)

    if attention_type == "linear"
        attention_layer = @compact(
            layer = LinearSpatialAttention(
                imsize=imsize,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        ) do x
            x, t = x
            @return layer(x)
        end
    elseif attention_type == "standard"
        attention_layer = @compact(
            layer = SpatialAttention(
                imsize=imsize,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        ) do x
            x, t = x
            @return layer(x)
        end
    elseif attention_type == "DiT"
        attention_layer = parameter_diffusion_transformer_block(
            in_channels=in_channels,
            out_channels=in_channels,
            pars_dim=t_pars_embedding_dims,
            embed_dim=embed_dim,
            number_heads=4,
            mlp_ratio=2,
            imsize=imsize,
            patch_size=(1, 1),
            number_patches=prod(div.(imsize, (1, 1)))
        )
    else
        attention_layer = Identity()
    end
    
    return attention_layer
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
# struct AttnParsConvNextUNet{L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13} <: Lux.AbstractExplicitContainerLayer{
#     (
#         :conv_in, :conv_out, :conv_down_blocks, 
#         :down_blocks, :bottleneck1, :bottleneck2, :conv_up_blocks, 
#         :up_blocks, :t_embedding, :pars_embedding, :attn_down_blocks,
#         :bottleneck_attn, :attn_up_blocks
#     )
# }
#     # layes::LayersType
#     conv_in::L1
#     conv_out::L2
#     conv_down_blocks::L3
#     down_blocks::L4
#     bottleneck1::L5
#     bottleneck2::L6
#     conv_up_blocks::L7
#     up_blocks::L8
#     t_embedding::L9
#     pars_embedding::L10
#     attn_down_blocks::L11
#     bottleneck_attn::L12
#     attn_up_blocks::L13
#     len_history::Int
# end
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


###############################################################################
# Conv Next UNet
###############################################################################

"""
    ConvNextUNet(
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
struct ConvNextUNet <: Lux.AbstractExplicitContainerLayer{
    (
        :conv_in, :init_conv_in, :conv_out, :conv_down_blocks, 
        :down_blocks, :bottleneck1, :bottleneck2, :conv_up_blocks, 
        :up_blocks, :t_embedding
    )
}
    conv_in::Lux.AbstractExplicitLayer
    init_conv_in::Lux.AbstractExplicitLayer
    conv_out::Lux.AbstractExplicitLayer
    conv_down_blocks::Lux.AbstractExplicitLayer
    down_blocks::Lux.AbstractExplicitLayer
    bottleneck1::Lux.AbstractExplicitLayer
    bottleneck2::Lux.AbstractExplicitLayer
    conv_up_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    t_embedding::Lux.AbstractExplicitLayer
end

function ConvNextUNet(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32,
    pars_dim=32
)

    multiplier = 2

    init_channels = div(channels[1], 2)

    conv_in = Conv((7, 7), (in_channels => init_channels); pad=3)
    init_conv_in = Conv((7, 7), (in_channels => init_channels); pad=3)

    # conv_out = Conv((1, 1), channels[1] => in_channels; init_weight=Lux.zeros32)

    t_embedding = Chain(
        x -> sinusoidal_embedding(x, min_freq, max_freq, embedding_dims),
        Lux.Dense(embedding_dims => embedding_dims),
        NNlib.gelu,
        Lux.Dense(embedding_dims => embedding_dims),
        NNlib.gelu,
    )


    conv_down_blocks = []
    down_blocks = []
    for i in 1:(length(channels) - 1)
        imsize = div.(image_size, 2^(i - 1))
        push!(conv_down_blocks, @compact(
            block_1 = conv_next_block(
                in_channels=channels[i], 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                pars_embed_dim=embedding_dims,
                imsize=imsize#div.(image_size, 2^(i - 1))
            ),
            block_2 = conv_next_block(
                in_channels=channels[i + 1], 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                pars_embed_dim=embedding_dims,
                imsize=imsize#div.(image_size, 2^(i - 1))
            )
            ) do x
                x, t = x
                x = block_1((x, t))
                @return block_2((x, t))
            end
        )


        push!(down_blocks, Conv(
            (4, 4), 
            (channels[i + 1] => channels[i + 1]); 
            use_bias=true,
            pad=1,
            stride=2
        ))
    end

    conv_down_blocks = Chain(conv_down_blocks...; disable_optimizations=true)
    down_blocks = Chain(down_blocks...; disable_optimizations=true)

    # # push!(down_blocks, DownBlock(channel_input, channels[1], block_depth))
    # for i in 1:(length(channels) - 1)
    #     Chain
    #     push!(down_blocks, ConvNextDownBlock(
    #         in_channels=channels[i], 
    #         out_channels=channels[i + 1], 
    #         block_depth=block_depth,
    #         multiplier=multiplier,
    #         pars_embed_dim=embedding_dims
    #     ))
    # end
    # down_blocks = Chain(down_blocks...; disable_optimizations=true)

    
    bottleneck1 = conv_next_block(
        in_channels=channels[end], 
        out_channels=channels[end],
        multiplier=multiplier,
        pars_embed_dim=embedding_dims,
        imsize=div.(image_size, 2^(length(channels) - 1))
    )
    bottleneck2 = conv_next_block(
        in_channels=channels[end], 
        out_channels=channels[end],
        multiplier=multiplier,
        pars_embed_dim=embedding_dims,
        imsize=div.(image_size, 2^(length(channels) - 1))
    )

    reverse!(channels)
    conv_up_blocks = []
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
        push!(conv_up_blocks, @compact(
            block_1 = conv_next_block(
                in_channels=channels[i] * 2, 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                pars_embed_dim=embedding_dims,
                imsize=imsize#div.(image_size, 2^(length(channels) - i - 1))
            ),
            block_2 = conv_next_block(
                in_channels=channels[i + 1], 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                pars_embed_dim=embedding_dims,
                imsize=imsize#div.(image_size, 2^(length(channels) - i - 1))
            )
            ) do x
                x, t = x
                x = block_1((x, t))
                @return block_2((x, t))
            end
        )
    end

    conv_up_blocks = Chain(conv_up_blocks...; disable_optimizations=true)
    up_blocks = Chain(up_blocks...; disable_optimizations=true)

    conv_out = @compact(
            ds_conv = Lux.Conv((7, 7), channels[end] => channels[end], pad=3),#, groups=channels[end]),
            conv_net = Chain(
                Lux.InstanceNorm(channels[end]),
                # Lux.BatchNorm(channels[end]),
                # Lux.GroupNorm(channels[end], 1),
                Lux.Conv((3, 3), (channels[end] => channels[end] * multiplier); pad=1),
                NNlib.gelu,
                Lux.InstanceNorm(channels[end] * multiplier),
                # Lux.BatchNorm(channels[end]),
                # Lux.GroupNorm(channels[end], 1),
                Lux.Conv((3, 3), (channels[end] * multiplier => channels[end]); pad=1),
            ),
            final = Conv((1, 1), (channels[end] => in_channels); use_bias=false)
        ) do x
            h = ds_conv(x)
            h = conv_net(h)
            x = h .+ x
            @return final(x)
        end

    return ConvNextUNet(
        conv_in, init_conv_in, conv_out, conv_down_blocks, 
        down_blocks, bottleneck1, bottleneck2, conv_up_blocks, 
        up_blocks, t_embedding
    )
end


function (conv_next_unet::ConvNextUNet)(
    x,#::Tuple{AbstractArray{T, 4}, AbstractArray{T, 4}, AbstractArray{S, 4}}, 
    ps::NamedTuple,
    st::NamedTuple
)# where T <: AbstractFloat


    x, x_0, t = x

    t_emb, new_st = conv_next_unet.t_embedding(t, ps.t_embedding, st.t_embedding)
    @set! st.t_embedding = new_st

    
    x, new_st = conv_next_unet.conv_in(x, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    x_0, new_st = conv_next_unet.init_conv_in(x_0, ps.init_conv_in, st.init_conv_in)
    @set! st.init_conv_in = new_st

    x = cat(x, x_0; dims=3)
    skips = (x, )
    for i in 1:length(conv_next_unet.conv_down_blocks)
        conv_layer_name = Symbol(:layer_, i)
        x, new_st = conv_next_unet.conv_down_blocks[i](
            (x, t_emb), ps.conv_down_blocks[conv_layer_name], st.conv_down_blocks[conv_layer_name]
        )  
        @set! st.conv_down_blocks[conv_layer_name] = new_st
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


    ########################
    # Add attention later
    ########################

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
        x, new_st = conv_next_unet.conv_up_blocks[i](
            (x, t_emb), ps.conv_up_blocks[layer_name], st.conv_up_blocks[layer_name]
        )
        @set! st.conv_up_blocks[layer_name] = new_st

        ########################
        # Add attention later
        ########################
        
    end

    x, new_st = conv_next_unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end


###############################################################################
# Parameter Diffusion Transformer Block
###############################################################################

"""
    parameter_diffusion_transformer_block(
        in_channels::Int,
        out_channels::Int,
        pars_dim::Int,
        embed_dim::Int,
        number_heads::Int,
        mlp_ratio::Int,
        imsize::Tuple{Int, Int},
        patch_size::Int,
        number_patches::Int
    )

Create a parameter diffusion transformer block with the given number of input
and output channels, parameter dimensions, embedding dimensions, number of heads,
MLP ratio, image size, patch size, and number of patches.
"""
function parameter_diffusion_transformer_block(;
    in_channels::Int,
    out_channels::Int,
    pars_dim::Int,
    embed_dim::Int,
    number_heads::Int,
    mlp_ratio::Int,
    imsize::Tuple{Int, Int},
    patch_size::Tuple{Int, Int},
    number_patches::Int
)
    @compact(
        norm = Lux.InstanceNorm(in_channels),

        pars_embedding = Lux.Chain(
            Lux.Dense(pars_dim => embed_dim),
        ),
        patchify_layer = patchify(
            imsize; 
            in_channels=in_channels, 
            patch_size=patch_size, 
            embed_planes=embed_dim
        ),
        positional_embedding = ViPosEmbedding(embed_dim, number_patches),
        dit_block = DiffusionTransformerBlock(embed_dim, number_heads, mlp_ratio),
        final_layer = FinalLayer(embed_dim, patch_size, out_channels), # (E, patch_size ** 2 * out_channels, N)
    ) do x
        x, pars = x

        pars = pars_embedding(pars)

        x = patchify_layer(x)
        x = positional_embedding(x)

        x = dit_block((x, pars))

        x = final_layer((x, pars))

        h = div(imsize[1], patch_size[1])
        w = div(imsize[2], patch_size[2])
        @return unpatchify(x, (h, w), patch_size, out_channels)
    end
end



function get_attention_block(
    attention_type::String, # "spatial" or "dit"
    in_channels::Int,
    out_channels::Int,
    pars_dim::Int,
    embed_dim::Int,
    number_heads::Int,
    mlp_ratio::Int,
    imsize::Tuple{Int, Int},
    patch_size::Tuple{Int, Int},
    number_patches::Int;
    dropout_rate=0.1f0,
)

    if attention_type == "dit"
        return parameter_diffusion_transformer_block(
            in_channels=channels[i + 1],
            out_channels=channels[i + 1],
            pars_dim=t_pars_embedding_dims,
            embed_dim=channels[i + 1],
            number_heads=number_heads,
            mlp_ratio=mlp_ratio,
            imsize=imsize,
            patch_size=(1, 1),
            number_patches=prod(div.(imsize, (1, 1)))
        )
    else
        return SpatialAttention(
            imsize=imsize,
            in_channels=channels[i + 1],
            embed_dim=embed_dim,
            num_heads=number_heads,
        )
    end
end







###############################################################################
# DiT Pars Conv Next UNet
###############################################################################

"""
DitParsConvNextUNet(
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
struct DitParsConvNextUNet <: Lux.AbstractExplicitContainerLayer{
    (
        :conv_in, :init_conv_in, :conv_out, :conv_down_blocks, 
        :down_blocks, :bottleneck1, :bottleneck2, :conv_up_blocks, 
        :up_blocks, :t_embedding, :pars_embedding, :dit_down_blocks,
        :bottleneck_dit, :dit_up_blocks
    )
}
    conv_in::Lux.AbstractExplicitLayer
    init_conv_in::Lux.AbstractExplicitLayer
    conv_out::Lux.AbstractExplicitLayer
    conv_down_blocks::Lux.AbstractExplicitLayer
    down_blocks::Lux.AbstractExplicitLayer
    bottleneck1::Lux.AbstractExplicitLayer
    bottleneck2::Lux.AbstractExplicitLayer
    conv_up_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    t_embedding::Lux.AbstractExplicitLayer
    pars_embedding::Lux.AbstractExplicitLayer
    dit_down_blocks::Lux.AbstractExplicitLayer
    bottleneck_dit::Lux.AbstractExplicitLayer
    dit_up_blocks::Lux.AbstractExplicitLayer
end

function DitParsConvNextUNet(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32,
    pars_dim=32,
    len_history=1,
    use_attention_in_layer=[false, false, true, true],
    padding="constant"
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

    t_embedding = Chain(
        x -> sinusoidal_embedding(x, min_freq, max_freq, t_pars_embedding_dims),
        Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
        NNlib.gelu,
        Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
        NNlib.gelu,
    )

    init_channels = div(channels[1], 2)

    # conv_in = Conv((7, 7), (in_channels => init_channels); pad=3)
    conv_in = Chain(
        get_padding(padding, 3),
        Conv((7, 7), (in_channels => init_channels))
    )

    # init_conv_in = Conv((7, 7), (len_history*in_channels => init_channels); pad=3)
    init_conv_in = Chain(
        get_padding(padding, 3),
        Conv((7, 7), (len_history*in_channels => init_channels))
    )

    conv_down_blocks = []
    dit_down_blocks = []
    down_blocks = []
    for i in 1:(length(channels) - 1)
        imsize = div.(image_size, 2^(i - 1))
        push!(
            conv_down_blocks, 
            multiple_conv_next_blocks(
                in_channels=channels[i], 
                out_channels=channels[i + 1], 
                multiplier=multiplier,
                embedding_dims=t_pars_embedding_dims,
                imsize=imsize
            )
        )

        if use_attention_in_layer[i]
            push!(dit_down_blocks, parameter_diffusion_transformer_block(
                in_channels=channels[i + 1],
                out_channels=channels[i + 1],
                pars_dim=t_pars_embedding_dims,
                embed_dim=channels[i + 1],
                number_heads=4,
                mlp_ratio=2,
                imsize=imsize,
                patch_size=(1, 1),
                number_patches=prod(div.(imsize, (1, 1)))
            ))
        else
            push!(dit_down_blocks, StateParsIdentity())
        end


        push!(down_blocks, Conv(
            (4, 4), 
            (channels[i + 1] => channels[i + 1]); 
            use_bias=true,
            pad=1,
            stride=2
        ))
    end

    conv_down_blocks = Chain(conv_down_blocks...; disable_optimizations=true)
    dit_down_blocks = Chain(dit_down_blocks...; disable_optimizations=true)
    down_blocks = Chain(down_blocks...; disable_optimizations=true)

    imsize = div.(image_size, 2^(length(channels) - 1))
    
    bottleneck1 = conv_next_block(
        in_channels=channels[end], 
        out_channels=channels[end],
        multiplier=multiplier,
        pars_embed_dim=t_pars_embedding_dims,
        imsize=imsize
    )


    bottleneck_dit = @compact(
            dit_1 = parameter_diffusion_transformer_block(
                in_channels=channels[end],
                out_channels=channels[end],
                pars_dim=t_pars_embedding_dims,
                embed_dim=channels[end],
                number_heads=4,
                mlp_ratio=2,
                imsize=imsize,
                patch_size=(1, 1),
                number_patches=prod(div.(imsize, (1, 1)))
            ) 
        ) do x
            x, pars = x
            @return dit_1((x, pars))
    end

    bottleneck2 = conv_next_block(
        in_channels=channels[end], 
        out_channels=channels[end],
        multiplier=multiplier,
        pars_embed_dim=t_pars_embedding_dims,
        imsize=imsize
    )

    reverse!(channels)
    conv_up_blocks = []
    dit_up_blocks = []
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
                embedding_dims=t_pars_embedding_dims,
                imsize=imsize
            )
        )

        if use_attention_in_layer[i]
            push!(dit_up_blocks, parameter_diffusion_transformer_block(
                in_channels=channels[i + 1],
                out_channels=channels[i + 1],
                pars_dim=t_pars_embedding_dims,
                embed_dim=channels[i + 1],
                number_heads=4,
                mlp_ratio=2,
                imsize=imsize,
                patch_size=(1, 1),
                number_patches=prod(div.(imsize, (1, 1)))
            ))
        else
            push!(dit_up_blocks, StateParsIdentity())
        end
    end

    conv_up_blocks = Chain(conv_up_blocks...; disable_optimizations=true)
    dit_up_blocks = Chain(dit_up_blocks...; disable_optimizations=true)
    up_blocks = Chain(up_blocks...; disable_optimizations=true)

    conv_out = @compact(
        ds_conv = Chain(
            get_padding(padding, 3),
            Lux.Conv((7, 7), channels[end] => channels[end])#, groups=in_channels)
        ),
        conv_net = Chain(
            Lux.InstanceNorm(channels[end]),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (channels[end] => channels[end] * multiplier)),
            NNlib.gelu,
            Lux.InstanceNorm(channels[end] * multiplier),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (channels[end] * multiplier => channels[end])),
        ),
        final = Conv((1, 1), (channels[end] => in_channels); use_bias=false)
    ) do x
        h = ds_conv(x)
        h = conv_net(h)
        x = h .+ x
        @return final(x)
    end

    return DitParsConvNextUNet(
        conv_in, init_conv_in, conv_out, conv_down_blocks, 
        down_blocks, bottleneck1, bottleneck2, conv_up_blocks, 
        up_blocks, t_embedding, pars_embedding, dit_down_blocks,
        bottleneck_dit, dit_up_blocks
    )
end


function (conv_next_unet::DitParsConvNextUNet)(
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

    t_pars_embed = pars_cat(t_emb, pars; dims=1)

    x, new_st = conv_next_unet.conv_in(x, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    x_0, new_st = conv_next_unet.init_conv_in(x_0, ps.init_conv_in, st.init_conv_in)
    @set! st.init_conv_in = new_st

    x = cat(x, x_0; dims=3)


    skips = (x, )
    for i in 1:length(conv_next_unet.conv_down_blocks)

        conv_layer_name = Symbol(:layer_, i)
        x, new_st = conv_next_unet.conv_down_blocks[i](
            (x, t_pars_embed), ps.conv_down_blocks[conv_layer_name], st.conv_down_blocks[conv_layer_name]
        )  
        @set! st.conv_down_blocks[conv_layer_name] = new_st

        x, new_st = conv_next_unet.dit_down_blocks[i](
            (x, t_pars_embed), ps.dit_down_blocks[conv_layer_name], st.dit_down_blocks[conv_layer_name]
        )
        @set! st.dit_down_blocks[conv_layer_name] = new_st

        skips = (skips..., x)

        x, new_st = conv_next_unet.down_blocks[i](
            x, ps.down_blocks[conv_layer_name], st.down_blocks[conv_layer_name]
        )
        @set! st.down_blocks[conv_layer_name] = new_st
    end

    x, new_st = conv_next_unet.bottleneck1(
        (x, t_pars_embed), ps.bottleneck1, st.bottleneck1
    )
    @set! st.bottleneck1 = new_st

    x, new_st = conv_next_unet.bottleneck_dit(
        (x, t_pars_embed), ps.bottleneck_dit, st.bottleneck_dit
    )
    @set! st.bottleneck_dit = new_st

    x, new_st = conv_next_unet.bottleneck2(
        (x, t_pars_embed), ps.bottleneck2, st.bottleneck2
    )
    @set! st.bottleneck2 = new_st

    for i in 1:length(conv_next_unet.conv_up_blocks)
        layer_name = Symbol(:layer_, i)

        x, new_st = conv_next_unet.up_blocks[i](
            x, ps.up_blocks[layer_name], st.up_blocks[layer_name]
        )
        @set! st.up_blocks[layer_name] = new_st


        x = cat(x, skips[end-i+1]; dims=3) # cat on channel  
              
        x, new_st = conv_next_unet.conv_up_blocks[i](
            (x, t_pars_embed), ps.conv_up_blocks[layer_name], st.conv_up_blocks[layer_name]
        )
        @set! st.conv_up_blocks[layer_name] = new_st
        
        x, new_st = conv_next_unet.dit_up_blocks[i](
            (x, t_pars_embed), ps.dit_up_blocks[layer_name], st.dit_up_blocks[layer_name]
        )
        @set! st.dit_up_blocks[layer_name] = new_st
        
    end

    x, new_st = conv_next_unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end



