using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA

"""
    DenoisingDiffusionProbabilisticModel(
        image_size::Tuple{Int, Int}; 
        in_channels::Int = 3,
        channels=[32, 64, 96, 128], 
        block_depth=2,
        min_freq=1.0f0, 
        max_freq=1000.0f0, 
        embedding_dims=32,
        timesteps::Int=100,
        init_beta::AbstractFloat=0.0001, 
        final_beta::AbstractFloat=0.02,
        type::String="linear"
    )
    

Creates a Denoising Diffusion Probabilistic Model (DDPM) with a UNet architecture.
"""
struct DenoisingDiffusionProbabilisticModel <: Lux.AbstractExplicitContainerLayer{
    (:unet, )
}
    unet::Lux.AbstractExplicitLayer
    noise_scheduling::NoiseScheduling
    sample::Function
    timesteps::Int
end

function DenoisingDiffusionProbabilisticModel(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32,
    timesteps::Int=100,
    init_beta::AbstractFloat=0.0001, 
    final_beta::AbstractFloat=0.02,
    type::String="linear"
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

    noise_scheduling = get_noise_scheduling(
        timesteps;
        init_beta=init_beta, 
        final_beta=final_beta,
        type=type
    )

    sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.sample(
        unet, 
        noise_scheduling,
        ps,
        st,
        rng,
        timesteps,
        num_samples, 
        dev
    )
    
    return DenoisingDiffusionProbabilisticModel(unet, noise_scheduling, sample, timesteps)
end