using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding

###############################################################################
# sinusoidal_embedding
###############################################################################
"""
    sinusoidal_embedding(
        x::AbstractArray{T, 4},
        min_freq::T,
        max_freq::T,
        embedding_dims::Int
    ) where {T <: AbstractFloat}

Embed the noise variances to a sinusoidal embedding with the given frequency
range and embedding dimensions.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function sinusoidal_embedding(
    x, 
    min_freq::T, 
    max_freq::T,
    embedding_dims::Int,
    dev=gpu
) where {T <: AbstractFloat}

    if size(x)[1:3] != (1, 1, 1)
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    # define frequencies
    # LinRange requires @adjoint when used with Zygote
    # Instead we manually implement range.
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> dev
    @assert length(freqs) == div(embedding_dims, 2)
    @assert size(freqs) == (div(embedding_dims, 2),)

    angular_speeds = reshape(convert(T, 2) * π * freqs, (1, 1, length(freqs), 1))
    @assert size(angular_speeds) == (1, 1, div(embedding_dims, 2), 1)

    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)
    @assert size(embeddings) == (1, 1, embedding_dims, size(x, 4))

    return embeddings
end


###############################################################################
# residual_block
###############################################################################

"""
    residual_block(
        in_channels::Int, 
        out_channels::Int,
        kernel_size::Tuple{Int, Int} = (3, 3),
    )

Create a residual block with the given number of input and output channels. 
The block consists of two convolutional layers with kernel size `kernel_size`.
The first layer has the same number of input and output channels, while the 
second layer has the same number of output channels as the block. 
The block also includes batch normalization and a skip connection.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function residual_block(
    in_channels::Int, 
    out_channels::Int,
    kernel_size::Tuple{Int, Int} = (3, 3)
)

    if in_channels == out_channels
        first_layer = NoOpLayer()
    else
        first_layer = Conv(kernel_size, in_channels => out_channels; pad=SamePad())
    end

    return Chain(
        first_layer,
        SkipConnection(
            Chain(
                BatchNorm(out_channels),
                a -> pad(a, :periodic, (1, 1)),
                Conv(kernel_size, out_channels => out_channels; stride=1),#, pad=(1, 1)), 
                swish,
                a -> pad(a, :periodic, (1, 1)),
                Conv(kernel_size, out_channels => out_channels; stride=1)#, pad=(1, 1))
            ), +
        )
    )
end

###############################################################################
# Downblock
###############################################################################

"""
    DownBlock(
        in_channels::Int, 
        out_channels::Int,
        block_depth::Int
    )

Create a down block with the given number of input and output channels and
block depth. The block consists of `block_depth` residual blocks followed by
a max pooling layer.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
struct DownBlock <: Lux.AbstractExplicitContainerLayer{
    (:residual_blocks, :maxpool)
}
    residual_blocks::Lux.AbstractExplicitLayer
    maxpool::MaxPool
end

function DownBlock(
    in_channels::Int, 
    out_channels::Int, 
    block_depth::Int
)
    layers = []

    push!(layers, residual_block(in_channels, out_channels))
    for _ in 1:block_depth
        push!(layers, residual_block(out_channels, out_channels))
    end

    # disable optimizations to keep block index
    residual_blocks = Chain(layers...; disable_optimizations=true)
    maxpool = MaxPool((2, 2); pad=0)

    return DownBlock(residual_blocks, maxpool)
end

function (db::DownBlock)(
    x::AbstractArray{T, 4}, 
    ps::NamedTuple,
    st::NamedTuple
) where {T <: AbstractFloat}

    skips = () # accumulate intermediate outputs
    for i in 1:length(db.residual_blocks)
        layer_name = Symbol(:layer_, i)
        x, new_st = db.residual_blocks[i](
            x, 
            ps.residual_blocks[layer_name],
            st.residual_blocks[layer_name]
        )
        # Don't use push! on vector because it invokes Zygote error
        skips = (skips..., x)
        @set! st.residual_blocks[layer_name] = new_st
    end
    x, _ = db.maxpool(x, ps.maxpool, st.maxpool)
    return (x, skips), st
end


###############################################################################
# Upblock
###############################################################################

"""
    UpBlock(
        in_channels::Int, 
        out_channels::Int,
        block_depth::Int
    )

Create an up block with the given number of input and output channels and
block depth. The block consists of `block_depth` residual blocks followed by
an upsampling layer.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
struct UpBlock <: Lux.AbstractExplicitContainerLayer{
    (:residual_blocks, :upsample)
}
    residual_blocks::Lux.AbstractExplicitLayer
    upsample::Upsample
end

function UpBlock(
    in_channels::Int, 
    out_channels::Int, 
    block_depth::Int
)

    layers = []
    push!(layers, residual_block(in_channels + out_channels, out_channels))
    for _ in 1:block_depth
        push!(layers, residual_block(out_channels * 2, out_channels))
    end
    residual_blocks = Chain(layers...; disable_optimizations=true)
    upsample = Upsample(:bilinear; scale=2)

    return UpBlock(residual_blocks, upsample)
end

function (up::UpBlock)(
    x, 
    ps,
    st::NamedTuple
)

    x, skips = x
    x, _ = up.upsample(x, ps.upsample, st.upsample)
    for i in 1:length(up.residual_blocks)
        layer_name = Symbol(:layer_, i)
        x = cat(x, skips[end - i + 1]; dims=3) # cat on channel
        x, new_st = up.residual_blocks[i](
            x, 
            ps.residual_blocks[layer_name],
            st.residual_blocks[layer_name]
        )
        @set! st.residual_blocks[layer_name] = new_st
    end

    return x, st
end

###############################################################################
# UNet
###############################################################################

"""
    UNet(
        image_size::Tuple{Int, Int},
        channels::Vector{Int} = [32, 64, 96, 128],
        block_depth::Int = 2,
        min_freq::Float32 = 1.0f0,
        max_freq::Float32 = 1000.0f0,
        embedding_dims::Int = 32
    )

Create a U-Net model with the given image size, number of channels, block depth,
frequency range, and embedding dimensions.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
struct UNet <: Lux.AbstractExplicitContainerLayer{
    (:upsample, :conv_in, :conv_out, :down_blocks, :residual_blocks, :up_blocks)
}
    upsample::Upsample
    conv_in::Conv
    conv_out::Conv
    down_blocks::Lux.AbstractExplicitLayer
    residual_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    noise_embedding::Function    
end

function UNet(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32
)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1), in_channels => channels[1])
    conv_out = Conv((1, 1), channels[1] => in_channels; init_weight=Lux.zeros32)

    noise_embedding = x -> sinusoidal_embedding(x, min_freq, max_freq, embedding_dims)

    channel_input = embedding_dims + channels[1]

    down_blocks = []
    push!(down_blocks, DownBlock(channel_input, channels[1], block_depth))
    for i in 1:(length(channels) - 2)
        push!(down_blocks, DownBlock(channels[i], channels[i + 1], block_depth))
    end
    down_blocks = Chain(down_blocks...; disable_optimizations=true)

    
    residual_blocks = []
    push!(residual_blocks, residual_block(channels[end - 1], channels[end]))
    for _ in 2:block_depth
        push!(residual_blocks, residual_block(channels[end], channels[end]))
    end
    residual_blocks = Chain(residual_blocks...; disable_optimizations=true)

    reverse!(channels)
    up_blocks = [UpBlock(channels[i], channels[i + 1], block_depth)
                 for i in 1:(length(channels) - 1)]
    up_blocks = Chain(up_blocks...)

    return UNet(upsample, conv_in, conv_out, down_blocks, residual_blocks, up_blocks, noise_embedding)
end


function (unet::UNet)(
    x::Tuple{AbstractArray{T, 4}, AbstractArray{S, 4}}, 
    ps::NamedTuple,
    st::NamedTuple
) where {T <: AbstractFloat, S <: Any}

    noisy_images, noise_variances = x
    @assert size(noise_variances)[1:3] == (1, 1, 1)
    @assert size(noisy_images, 4) == size(noise_variances, 4)

    emb = unet.noise_embedding(noise_variances)
    @assert size(emb)[[1, 2, 4]] == (1, 1, size(noise_variances, 4))
    emb, _ = unet.upsample(emb, ps.upsample, st.upsample)
    @assert size(emb)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noise_variances, 4))

    x, new_st = unet.conv_in(noisy_images, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    x = cat(x, emb; dims=3)
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    skips_at_each_stage = ()
    for i in 1:length(unet.down_blocks)
        layer_name = Symbol(:layer_, i)
        (x, skips), new_st = unet.down_blocks[i](x, ps.down_blocks[layer_name],
                                                 st.down_blocks[layer_name])
        #x = leakyrelu.(x)                                              
        @set! st.down_blocks[layer_name] = new_st
        skips_at_each_stage = (skips_at_each_stage..., skips)
    end

    x, new_st = unet.residual_blocks(x, ps.residual_blocks, st.residual_blocks)
    @set! st.residual_blocks = new_st

    for i in 1:length(unet.up_blocks)
        layer_name = Symbol(:layer_, i)
        x, new_st = unet.up_blocks[i]((x, skips_at_each_stage[end - i + 1]),
                                      ps.up_blocks[layer_name], st.up_blocks[layer_name])
        #x = leakyrelu.(x)
        @set! st.up_blocks[layer_name] = new_st
    end

    x, new_st = unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end
struct ConditionalUNet <: Lux.AbstractExplicitContainerLayer{
    (:upsample, :conv_in, :init_conv_in, :conv_out, :down_blocks, :residual_blocks, :up_blocks)
}
    upsample::Upsample
    conv_in::Conv
    init_conv_in::Conv
    conv_out::Conv
    down_blocks::Lux.AbstractExplicitLayer
    residual_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    noise_embedding::Function
end

function ConditionalUNet(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32
)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1), in_channels => embedding_dims)
    init_conv_in = Conv((1, 1), in_channels => embedding_dims)
    conv_out = Conv((1, 1), channels[1] => in_channels; init_weight=Lux.zeros32)

    noise_embedding = x -> sinusoidal_embedding(x, min_freq, max_freq, embedding_dims)

    channel_input = embedding_dims + embedding_dims + embedding_dims #channels[1] + channels[1]

    down_blocks = []
    push!(down_blocks, DownBlock(channel_input, channels[1], block_depth))
    for i in 1:(length(channels) - 2)
        push!(down_blocks, DownBlock(channels[i], channels[i + 1], block_depth))
    end
    down_blocks = Chain(down_blocks...; disable_optimizations=true)

    
    residual_blocks = []
    push!(residual_blocks, residual_block(channels[end - 1], channels[end]))
    for _ in 2:block_depth
        push!(residual_blocks, residual_block(channels[end], channels[end]))
    end
    residual_blocks = Chain(residual_blocks...; disable_optimizations=true)

    reverse!(channels)
    up_blocks = [UpBlock(channels[i], channels[i + 1], block_depth)
                 for i in 1:(length(channels) - 1)]
    up_blocks = Chain(up_blocks...)

    return ConditionalUNet(upsample, conv_in, init_conv_in, conv_out, down_blocks, residual_blocks, up_blocks, noise_embedding)
end


function (conditional_unet::ConditionalUNet)(
    x::Tuple{AbstractArray{T, 4}, AbstractArray{T, 4}, AbstractArray{S, 4}}, 
    ps::NamedTuple,
    st::NamedTuple
) where {T <: AbstractFloat, S <: Any}

    noisy_images, init_noisy_images, noise_variances = x
    @assert size(noise_variances)[1:3] == (1, 1, 1)
    @assert size(noisy_images, 4) == size(noise_variances, 4)
    @assert size(init_noisy_images, 4) == size(noise_variances, 4)

    emb = conditional_unet.noise_embedding(noise_variances)
    @assert size(emb)[[1, 2, 4]] == (1, 1, size(noise_variances, 4))
    emb, _ = conditional_unet.upsample(emb, ps.upsample, st.upsample)
    @assert size(emb)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noise_variances, 4))

    x, new_st = conditional_unet.conv_in(noisy_images, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    init_x, new_st = conditional_unet.init_conv_in(init_noisy_images, ps.init_conv_in, st.init_conv_in)
    @set! st.init_conv_in = new_st
    @assert size(init_x)[[1, 2, 4]] ==
            (size(init_noisy_images, 1), size(init_noisy_images, 2), size(init_noisy_images, 4))

    x = cat(x, init_x, emb; dims=3)
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    skips_at_each_stage = ()
    for i in 1:length(conditional_unet.down_blocks)
        layer_name = Symbol(:layer_, i)
        (x, skips), new_st = conditional_unet.down_blocks[i](x, ps.down_blocks[layer_name],
                                                 st.down_blocks[layer_name])
        #x = leakyrelu.(x)                                              
        @set! st.down_blocks[layer_name] = new_st
        skips_at_each_stage = (skips_at_each_stage..., skips)
    end

    x, new_st = conditional_unet.residual_blocks(x, ps.residual_blocks, st.residual_blocks)
    @set! st.residual_blocks = new_st

    for i in 1:length(conditional_unet.up_blocks)
        layer_name = Symbol(:layer_, i)
        x, new_st = conditional_unet.up_blocks[i]((x, skips_at_each_stage[end - i + 1]),
                                      ps.up_blocks[layer_name], st.up_blocks[layer_name])
        #x = leakyrelu.(x)
        @set! st.up_blocks[layer_name] = new_st
    end

    x, new_st = conditional_unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end