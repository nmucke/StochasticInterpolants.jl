using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding
using LinearAlgebra
using Boltz
# import Boltz.VisionTransformerEncoder
import Boltz.MultiHeadAttention as MultiHeadAttention


###############################################################################
# sinusoidal_embedding
###############################################################################
"""
    sinusoidal_embedding(
        x::AbstractArray{AbstractFloat, 4},
        min_freq::AbstractFloat,
        max_freq::AbstractFloat,
        embedding_dims::Int
    )

Embed the noise variances to a sinusoidal embedding with the given frequency
range and embedding dimensions.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function sinusoidal_embedding(
    x, 
    min_freq::AbstractFloat, 
    max_freq::AbstractFloat,
    embedding_dims::Int,
    dev=gpu_device()
)

    if length(size(x)) != 4
        x = reshape(x, (1, 1, 1, size(x)[end]))
        # throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    # define frequencies
    # LinRange requires @adjoint when used with Zygote
    # Instead we manually implement range.
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> dev
    # @assert length(freqs) == div(embedding_dims, 2)
    # @assert size(freqs) == (div(embedding_dims, 2),)

    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))
    # @assert size(angular_speeds) == (1, 1, div(embedding_dims, 2), 1)

    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)
    # @assert size(embeddings) == (1, 1, embedding_dims, size(x, 4))

    return dropdims(embeddings, dims=(1, 2)) #embeddings
end

