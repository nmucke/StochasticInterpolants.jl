using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding
using LinearAlgebra
using GPUArraysCore
# using Boltz
# import Boltz.VisionTransformerEncoder
# import Boltz.MultiHeadAttention as MultiHeadAttention

"""
    position_encoding(dim_embedding::Int, max_length::Int=1000)

Create a position encoding for a transformer model.

Based on https://liorsinai.github.io/machine-learning/2022/05/18/transformers.html#position-encodings
"""
struct PositionEncoding{W <: AbstractArray}
    weight::W
end

function PositionEncoding(dim_embedding::Int, max_length::Int=1000)
    W = make_position_encoding(dim_embedding, max_length)
    PositionEncoding(W)
end

function make_position_encoding(dim_embedding::Int, seq_length::Int, n::Int=10000)
    encoding = Matrix{Float32}(undef, dim_embedding, seq_length)
    for pos in 1:seq_length
        for row in 0:2:(dim_embedding - 1)
            denom = 1/(n^(row/dim_embedding))
            encoding[row + 1, pos] = sin(pos * denom)
            encoding[row + 2, pos] = cos(pos * denom)
        end
    end
    encoding    
end

function Base.show(io::IO, pe::PositionEncoding)
    print(io, "PositionEncoding($(size(pe.weight, 1)))")
end

(pe::PositionEncoding)(x::AbstractArray) = (pe::PositionEncoding)(size(x, 2))
function (pe::PositionEncoding)(seq_length::Int)
    max_length = size(pe.weight, 2)
    if seq_length > max_length
        error("sequence length of $seq_length exceeds maximum position encoding length of $max_length")
    end
    pe.weight[:, Base.OneTo(seq_length)]
end




"""
    _flatten_spatial(x::AbstractArray{T, 4})

Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3)
"""
@inline function _flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

"""
    _fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})

Type-stable and faster version of `MLUtils.chunk`.
"""
@inline _fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function _fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, _fast_chunk(h, n))
end
@inline function _fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    return _fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end
@inline function _fast_chunk(
        x::GPUArraysCore.AnyGPUArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return copy(selectdim(x, dim, _fast_chunk(h, n)))
end

"""
    MultiHeadSelfAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0)

Multi-head self-attention layer

## Arguments

  - `planes`: number of input channels
  - `nheads`: number of heads
  - `qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attn_dropout_prob`: dropout probability after the self-attention layer
  - `proj_dropout_prob`: dropout probability after the projection layer
"""
function MultiHeadSelfAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0) where {T}
    # @argcheck in_planes % number_heads == 0

    qkv_layer = Lux.Dense(in_planes, in_planes * 3; use_bias=qkv_bias)
    attention_dropout = Lux.Dropout(attention_dropout_rate)
    projection = Lux.Chain(
        Lux.Dense(in_planes => in_planes), Lux.Dropout(projection_dropout_rate))

    return Lux.@compact(; number_heads, qkv_layer, attention_dropout,
        projection, dispatch=:MultiHeadSelfAttention) do x::AbstractArray{<:Real, 3}
        qkv = qkv_layer(x)
        q, k, v = _fast_chunk(qkv, Val(3), Val(1))
        y, _ = NNlib.dot_product_attention(
            q, k, v; fdrop=attention_dropout, nheads=number_heads)
        @return projection(y)
    end
end


###############################################################################
# Linear attention
###############################################################################

# Copied from NNLib
split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)

"""
    LinearMultiHeadSelfAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0)

Linear Multi-head self-attention layer based on the paper "Efficient Attention: Attention with Linear Complexities"
"""
function LinearMultiHeadSelfAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, projection_dropout_rate::T=0.0f0) where {T}
    # @argcheck in_planes % number_heads == 0

    @compact(

        projection = Lux.Chain(
            Lux.Dense(in_planes => in_planes * 3; use_bias=false),
            Lux.Dropout(projection_dropout_rate)
        ),

        scale = 1 / sqrt(Float32(in_planes))
        
    ) do x

        x = projection(x) # (C * H, N, B) -> (C * H * 3, N, B)

        q, k, v = _fast_chunk(x, Val(3), Val(1)) # (C * 3 * H, N, B) -> (C * H, N, B), (C * H, N, B), (C * H, N, B)
        # q, k, v = x # (C, H, N, B)

        q, k, v = split_heads.((q, k, v), number_heads) # (C * H, N, B) -> (C, H, N, B), (C, H, N, B), (C, H, N, B)

        q = softmax(q, dims=1) # softmax over the channels
        k = softmax(k, dims=3) # softmax over the sequence

        q = q .* scale

        v = permutedims(v, (3, 1, 2, 4)) # (C, H, N, B) -> (N, C, H, B)
        k = permutedims(k, (1, 3, 2, 4)) # (C, H, N, B) -> (C, N, H, B)
        
        x = batched_mul(k, v) # (C, N, H, B) * (N, C, H, B) -> (C, C, H, B)

        q = permutedims(q, (3, 1, 2, 4)) # (C, H, N, B) -> (N, C, H, B)
        
        x = batched_mul(q, x) # (N, C, H, B) * (C, C, H, B) -> (N, C, H, B)

        x = permutedims(x, (2, 3, 1, 4)) # (N, C, H, B) -> (C, H, N, B)

        @return join_heads(x) # (C, H, N, B) -> (C * H, N, B)

    end
end


function LinearSpatialAttention(;
    imsize,
    in_channels,
    embed_dim,
    num_heads=1,
    dropout_rate=0.1f0,
)
    @compact(
        # norm = Lux.LayerNorm((in_channels, 1); affine=true),
        _patchify = patchify(
            imsize; 
            in_channels=in_channels, 
            patch_size=(1, 1), 
            embed_planes=embed_dim * num_heads
        ),
        position_encoding = convert(Array{Float32}, make_position_encoding(embed_dim * num_heads, imsize[1]*imsize[2])),
        norm_1 = Lux.LayerNorm((embed_dim * num_heads, imsize[1]*imsize[2])),
        attn = LinearMultiHeadSelfAttention(
            embed_dim * num_heads,
            num_heads; 
            attention_dropout_rate=dropout_rate,
            projection_dropout_rate=dropout_rate
        ),
        norm_2 = Lux.LayerNorm((embed_dim * num_heads, imsize[1]*imsize[2])),
        conv_out = Lux.Chain(
            # Lux.Dense(embed_dim * num_heads => embed_dim * num_heads),
            Lux.Dense(embed_dim * num_heads => in_channels),
            # NNlib.gelu,
            # Lux.LayerNorm((embed_dim * num_heads, imsize[1]*imsize[2])),
            Lux.LayerNorm((in_channels, imsize[1]*imsize[2])),
            # Lux.Dropout(dropout_rate),
            # Lux.Dense(embed_dim * num_heads => in_channels),
        ),
        # skip_conv = Lux.Dense(embed_dim * num_heads => in_channels)

        
    ) do x
        x = _patchify(x) # (Nx, Ny, C, B) -> (E * H, Nx*Ny, B)

        x = x .+ position_encoding

        # x_skip = x

        x = norm_1(x)

        x = attn(x)

        # x += x_skip

        # x = norm_2(x)

        # x_skip = x

        x = conv_out(x) # (E * H, Nx*Ny, B) -> (C, Nx*Ny, B)

        # x = x + skip_conv(x_skip)

        x = unpatchify(x, imsize, (1, 1), in_channels) # (C, Nx*Ny, B) - > (Nx, Ny, C, B)

        @return x

    end
end




###############################################################################
# Diffusion Transformer
###############################################################################

"""
    modulate(x, scale, shift)

Modulate the input tensor `x` by scaling and shifting it.
"""

function modulate(x, scale, shift)

    # Scale
    x = x .* (1.0f0 .+ scale)

    # Shift
    x = x .+ shift    

    return x
end

"""
    patchify(
        imsize::Tuple{Int, Int}=(64, 64),
        in_channels::Int=3,
        patch_size::Tuple{Int, Int}=(8, 8),
        embed_planes::Int=128,
        norm_layer=Returns(Lux.NoOpLayer()),
        flatten=true
    )

Create a patch embedding layer with the given image size, number of input
channels, patch size, embedding planes, normalization layer, and flatten flag.

Based on https://github.com/LuxDL/Boltz.jl/blob/v0.3.9/src/vision/vit.jl#L48-L61
"""
function patchify(
    imsize::Tuple{Int, Int}; 
    in_channels=3, 
    patch_size=(8, 8),
    embed_planes=128, 
    norm_layer=Returns(Lux.NoOpLayer()), 
    flatten=true
)

    im_width, im_height = imsize
    patch_width, patch_height = patch_size

    @assert (im_width % patch_width == 0) && (im_height % patch_height == 0)

    return Lux.Chain(
        Lux.Conv(patch_size, in_channels => embed_planes; stride=patch_size),
        flatten ? _flatten_spatial : identity, 
        norm_layer(embed_planes)
    )
end

"""
    unpatchify(x, patch_size, out_channels)

Unpatchify the input tensor `x` with the given patch size and number of output
channels.
"""
function unpatchify(x, imsize, patch_size, out_channels)
    
    c = out_channels
    p1, p2 = patch_size
    h, w = imsize
    @assert h * w == size(x, 2)

    x = reshape(x, (p1, p2, c, h, w, size(x, 3)))
    x = permutedims(x, (1, 4, 2, 5, 3, 6))
    imgs = reshape(x, (h * p1, w * p2, c, size(x, 6)))
    return imgs
end

function SpatialAttention(;
    imsize,
    in_channels,
    embed_dim,
    num_heads=1,
    dropout_rate=0.1f0,
)
    @compact(
        # norm = Lux.LayerNorm((in_channels, 1); affine=true),
        norm = Lux.InstanceNorm(in_channels),
        _patchify = patchify(
            imsize; 
            in_channels=in_channels, 
            patch_size=(1, 1), 
            embed_planes=embed_dim
        ),
        attn = MultiHeadSelfAttention(
            embed_dim, 
            num_heads; 
            attention_dropout_rate=dropout_rate,
            projection_dropout_rate=dropout_rate
        ),
        conv_out = Lux.Chain(
            Lux.Conv((1, 1), embed_dim => in_channels),
            Lux.InstanceNorm(in_channels),
            # Lux.LayerNorm((in_channels, 1); affine=true),
        )

        
    ) do x
        x_in = x
        x = norm(x)
        x = _patchify(x)
        x = attn(x)
        x = unpatchify(x, imsize, (1, 1), embed_dim)
        x = conv_out(x)
        @return x + x_in
    end
end








"""
    VisionTransformerEncoder(in_planes, depth, number_heads; mlp_ratio = 4.0f0,
        dropout = 0.0f0)

Transformer as used in the base ViT architecture.

## Arguments

  - `in_planes`: number of input channels
  - `depth`: number of attention blocks
  - `number_heads`: number of attention heads

## Keyword Arguments

  - `mlp_ratio`: ratio of MLP layers to the number of input channels
  - `dropout_rate`: dropout rate

## References

[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image
recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
"""
function VisionTransformerEncoder(
        in_planes, depth, number_heads; mlp_ratio=4.0f0, dropout_rate=0.0f0)
    hidden_planes = floor(Int, mlp_ratio * in_planes)
    layers = [Lux.Chain(
                  Lux.SkipConnection(
                      Lux.Chain(Lux.LayerNorm((in_planes, 1); affine=true),
                        MultiHeadAttention(
                              in_planes, number_heads; attention_dropout_rate=dropout_rate,
                              projection_dropout_rate=dropout_rate)),
                      +),
                  Lux.SkipConnection(
                      Lux.Chain(Lux.LayerNorm((in_planes, 1); affine=true),
                          Lux.Chain(Lux.Dense(in_planes => hidden_planes, NNlib.gelu),
                              Lux.Dropout(dropout_rate),
                              Lux.Dense(hidden_planes => in_planes),
                              Lux.Dropout(dropout_rate));
                          disable_optimizations=true),
                      +)) for _ in 1:depth]
    return Lux.Chain(layers...; disable_optimizations=true)
end


function reshape_modulation(x, seq_len)

    x = reshape(x, size(x)[1], 1, size(x)[2])
    x = repeat(x, 1, seq_len, 1)

    return x
end


