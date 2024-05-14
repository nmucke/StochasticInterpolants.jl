using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA
using DifferentialEquations

"""
StochasticInterpolant(
        image_size::Tuple{Int, Int}; 
        in_channels::Int = 3,
        channels=[32, 64, 96, 128], 
        block_depth=2,
        min_freq=1.0f0, 
        max_freq=1000.0f0, 
        embedding_dims=32,
        eps=1e-5,
    ) where T <: AbstractFloat
    

Creates a Denoising Diffusion Probabilistic Model (DDPM) with a UNet architecture.
"""
struct StochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:unet, )
}
    unet::UNet
    marginal_probability_std::Function
    diffusion_coefficient::Function
    sde_sample::Function
    ode_sample::Function
    eps::Float32
end

function StochasticInterpolant(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    sigma::AbstractFloat=25.0,
    min_freq=1.0f0,
    max_freq=1000.0f0,
    embedding_dims=32,
    eps=1e-5,
    num_steps=1000,
)

    unet = UNet(
        image_size; 
        in_channels=in_channels,
        channels=channels, 
        block_depth=block_depth,
        min_freq=min_freq, 
        max_freq=max_freq, 
        embedding_dims=embedding_dims,
    )

    # forward_drift_term(x, ps, t, st) = zeros(size(x))
    # forward_diffusion_term(x, ps, t, st) = sigma .^ t .* ones(size(x))

    sigma = Float32(sigma)
    marginal_probability_std(t) = sqrt.((sigma.^(2 .* t) .- 1) ./ 2 ./ log.(sigma))
    diffusion_coefficient(t) = sigma .^ t

    sde_sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.sde_sampler(
        unet,
        ps,
        st,
        rng,
        marginal_probability_std,
        diffusion_coefficient,
        num_samples,
        num_steps,
        eps,
        dev
    )

    ode_sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.ode_sampler(
        unet,
        ps,
        st,
        rng,
        marginal_probability_std,
        diffusion_coefficient,
        num_samples,
        num_steps,
        eps,
        dev
    )
    
    return StochasticInterpolant(
        unet, marginal_probability_std, diffusion_coefficient, sde_sample, ode_sample, eps
    )
end