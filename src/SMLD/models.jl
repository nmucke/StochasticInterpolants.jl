using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA
using DifferentialEquations


# sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.euler_maruyama_sampler(
#     unet,
#     ps,
#     st,
#     rng,
#     marginal_probability_std,
#     diffusion_coefficient,
#     num_samples,
#     num_steps,
#     eps,
#     dev
# )


# function sde_solver(
#     unet,
#     ps,
#     st,
#     rng,
#     marginal_probability_std,
#     diffusion_coefficient,
#     backward_drift_term,
#     backward_diffusion_term,
#     num_samples,
#     num_steps,
#     eps,
#     dev
# )

#     t = ones((1, 1, 1, num_samples)) |> dev
#     x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples) |> dev
#     x = x .* Float32.(marginal_probability_std(t))
#     timesteps = LinRange(1, eps, num_steps)# |> dev
#     step_size = Float32.(timesteps[1] - timesteps[2]) |> dev

#     problem = SDEProblem(
#         backward_drift_term, 
#         backward_diffusion_term, 
#         randn(rng, Float32, size(unet.upsample.size..., unet.conv_in.in_chs, num_samples)) .* marginal_probability_std(0.0), 
#         (1.0, 0.0)
#     )

#     dudt(u, ps, t, st) = backward_drift_term((u, 1-t), ps, st)
#     g(u, ps, t) = backward_diffusion_term(u, ps, 1-t, st)
#     prob = SDEProblem(dudt, g, x, tspan, nothing)


#     return x



"""
    marginal_probability_std(
        t::AbstractArray,
        sigma::AbstractFloat,
    )

Computes the standard deviation of the marginal probability of the noise

Based on https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=LZC7wrOvxLdL
"""
# function _marginal_probability_std(
#     t::AbstractArray,
#     sigma::AbstractFloat,
# )

#     return sqrt.((sigma.^(2 .* t) .- 1) ./ 2 ./ log.(sigma))
# end

"""
    diffusion_coefficient(
        t::AbstractArray,
        sigma::AbstractFloat,
    )

Computes the diffusion coefficient of the noise

Based on https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=LZC7wrOvxLdL
"""
# function _diffusion_coefficient(
#     t::AbstractArray,
#     sigma::AbstractFloat,
# )
#     return sigma.^t
# end



"""
ScoreMatchingLangevinDynamics(
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
struct ScoreMatchingLangevinDynamics <: Lux.AbstractExplicitContainerLayer{
    (:unet, )
}
    unet::UNet
    marginal_probability_std::Function
    diffusion_coefficient::Function
    sde_sample::Function
    ode_sample::Function
    eps::Float32
end

function ScoreMatchingLangevinDynamics(
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
    
    sde_sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.smld_sde_sampler(
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

    ode_sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.smld_ode_sampler(
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
    
    return ScoreMatchingLangevinDynamics(
        unet, marginal_probability_std, diffusion_coefficient, sde_sample, ode_sample, eps
    )
end