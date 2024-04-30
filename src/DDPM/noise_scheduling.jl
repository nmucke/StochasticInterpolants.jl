using Lux
using Random
using CUDA
using NNlib
using Setfield

"""
get_beta_schedule

Returns a schedule of beta values for the diffusion process.

Based on https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=qWw50ui9IZ5q
"""
function get_beta_schedule(
    timesteps::Int,
    init_beta::T=0.0001, 
    final_beta::T=0.02,
    type::String="linear"
) where {T <: AbstractFloat}

    if type == "linear"
        return range(init_beta, final_beta; length=timesteps)
    else
        throw(ArgumentError("Invalid type."))
    end
end


function forward_diffusion_sample(
    x_0,
    t,
    rng,
    noise_scheduling,
    dev,
)

    image_shape = size(x_0)

    noise = randn(rng, image_shape...) |> dev

    #CUDA.@allowscalar sqrt_alphas_cumprod_t = noise_scheduling.sqrt_alphas_cumprod[t]
    #CUDA.@allowscalar sqrt_one_minus_alphas_cumprod_t = noise_scheduling.sqrt_one_minus_alphas_cumprod[t]

    sqrt_alphas_cumprod_t = reshape(noise_scheduling.sqrt_alphas_cumprod[t], (1, 1, 1, size(x_0)[end]))
    sqrt_one_minus_alphas_cumprod_t = reshape(noise_scheduling.sqrt_one_minus_alphas_cumprod[t], (1, 1, 1, size(x_0)[end]))

    
    return sqrt_alphas_cumprod_t .* x_0 + sqrt_one_minus_alphas_cumprod_t .* noise, noise
end


"""
    NoiseScheduling

A struct to hold the noise scheduling parameters.

This struct holds the parameters for the noise scheduling process.

# Fields
- `betas::Vector{T}`: The beta values.
- `alphas::Vector{T}`: The alpha values.
- `alphas_cumprod::Vector{T}`: The cumulative product of the alpha values.
- `alphas_cumprod_prev::Vector{T}`: The cumulative product of the alpha values, shifted by one.
- `sqrt_recip_alphas::Vector{T}`: The square root of the reciprocal of the alpha values.
- `sqrt_alphas_cumprod::Vector{T}`: The square root of the cumulative product of the alpha values.
- `sqrt_one_minus_alphas_cumprod::Vector{T}`: The square root of one minus the cumulative product of the alpha values.
- `posterior_variance::Vector{T}`: The posterior variance.
"""
struct NoiseScheduling
    betas
    alphas
    alphas_cumprod
    alphas_cumprod_prev
    sqrt_recip_alphas
    sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod
    posterior_variance
end


function get_noise_scheduling(
    timesteps, 
    init_beta, 
    final_beta,
    type,
    dev=gpu
)

    betas = collect(get_beta_schedule(timesteps, init_beta, final_beta, type)) |> dev

    alphas = 1.0 .- betas |> dev

    alphas_cumprod = cumprod(alphas) |> dev
    CUDA.@allowscalar alphas_cumprod_prev = vcat([1.0], alphas_cumprod[1:end-1]) |> dev

    sqrt_recip_alphas = sqrt.(1.0 ./ alphas) |> dev
    sqrt_alphas_cumprod = sqrt.(alphas_cumprod) |> dev
    sqrt_one_minus_alphas_cumprod = sqrt.(1.0 .- alphas_cumprod) |> dev

    posterior_variance = betas .* (1. .- alphas_cumprod_prev) ./ (1. .- alphas_cumprod) |> dev
    
    return NoiseScheduling(
        betas,
        alphas,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        posterior_variance,
    )
end
