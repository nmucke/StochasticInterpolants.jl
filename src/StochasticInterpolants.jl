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

# Training
include("training.jl")
export train_diffusion_model

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
include("DDPM/noise_scheduling.jl")
include("DDPM/sampling.jl")
include("DDPM/models.jl")
include("DDPM/loss.jl")

# Noise scheduling
export get_beta_schedule
export get_index_from_list
export forward_diffusion_sample
export get_noise_scheduling
export NoiseScheduling

# Sampling
export sample_timestep
export sample

# Models
export DenoisingDiffusionProbabilisticModel

# Loss
export get_loss

##### SMLD #####
include("SMLD/models.jl")
include("SMLD/SDE.jl")
include("SMLD/loss.jl")
include("SMLD/sampling.jl")


# Noise scheduling
export marginal_probability_std
export diffusion_coefficient

# Loss
export get_loss

# Models
export ScoreMatchingLangevinDynamics

# Sampling
export euler_maruyama_sampler


end
