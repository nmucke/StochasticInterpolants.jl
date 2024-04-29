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

"""
    get_index_from_list

Returns the value of a list at a given index.

This function is used to get the value of a list at a given index, 
where the index is a tensor of batch indices.
"""
function get_index_from_list(
    vals, 
    t, 
    x_shape
)
    batch_size = size(t)[1]
    out = vals[t]
    
    return reshape(out, (batch_size, (1,) * (length(x_shape) - 1)))
end


"""
    forward_diffusion_sample

Takes an image and a timestep as input and returns the noisy version of it.
"""
function forward_diffusion_sample(
    x_0,
    t,
    rng,
    noise_scheduling,
    dev,
)

    image_shape = size(x_0)

    noise = randn(rng, image_shape...) |> dev
    sqrt_alphas_cumprod_t = noise_scheduling.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alphas_cumprod_t = noise_scheduling.sqrt_one_minus_alphas_cumprod[t]

    sqrt_alphas_cumprod_t = reshape(sqrt_alphas_cumprod_t, (1, 1, 1, size(x_0)[end]))
    sqrt_one_minus_alphas_cumprod_t = reshape(sqrt_one_minus_alphas_cumprod_t, (1, 1, 1, size(x_0)[end]))
    
    return sqrt_alphas_cumprod_t .* x_0 + sqrt_one_minus_alphas_cumprod_t .* noise, noise
end


"""
    NoiseScheduling

A struct to hold the noise scheduling parameters.
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
    type
)

    betas = get_beta_schedule(timesteps, init_beta, final_beta, type)

    alphas = 1.0 .- betas

    alphas_cumprod = cumprod(alphas)
    alphas_cumprod_prev = vcat([1.0], alphas_cumprod[1:end-1])

    sqrt_recip_alphas = sqrt.(1.0 ./ alphas)
    sqrt_alphas_cumprod = sqrt.(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = sqrt.(1.0 .- alphas_cumprod)

    posterior_variance = betas .* (1. .- alphas_cumprod_prev) ./ (1. .- alphas_cumprod)
    
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
