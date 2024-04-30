"""
    StochasticInterpolants

Implementation of the Stochastic Interpolants method for generative modeling.
"""
module StochasticInterpolants

using Lux
using Random
using CUDA
using NNlib
using Setfield
using LuxCUDA

include("neural_network_layers.jl")
export residual_block
export UNet
export UpBlock, DownBlock
export sinusoidal_embedding

##### DDIM #####
# Models
include("DDIM/models.jl")
export DenoisingDiffusionImplicitModel
export diffusion_schedules
export denoise
export reverse_diffusion
export denormalize
export generate

# Training
include("DDIM/training.jl")
export compute_loss
export train_step
export train_DDIM

##### DDPM #####
# Noise scheduling
include("DDPM/noise_scheduling.jl")
export get_beta_schedule
export get_index_from_list
export forward_diffusion_sample
export get_noise_scheduling
export NoiseScheduling

# Loss
include("DDPM/loss.jl")
export get_loss

# Sampling
include("DDPM/sampling.jl")
export sample_timestep








end
