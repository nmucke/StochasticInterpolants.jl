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
using Boltz


##### Layers #####
include("neural_network_layers/conv_next.jl")
include("neural_network_layers/embeddings.jl")
include("neural_network_layers/transformer.jl")
include("neural_network_layers/conv.jl")
export residual_block
export conv_next_block
export ConvNextDownBlock
export ConvNextUpBlock
export UNet
export ConditionalUNet
export UpBlock, DownBlock
export sinusoidal_embedding
export modulate
export patchify
export unpatchify
export reshape_modulation
export DiffusionTransformerBlock
# export VisionTransformerEncoder
export FinalLayer
export DiffusionTransformer
export ConditionalDiffusionTransformer
export ConvNextUNet
export parameter_diffusion_transformer_block
# export ParsConvNextUNet
export SpatialAttention
export DitParsConvNextUNet
export AttnParsConvNextUNet


include("unet_transformer.jl")
export dit_down_block
export dit_up_block


##### SI #####
include("SI/interpolants.jl")
include("SI/models.jl")
include("SI/conditional_models.jl")
include("SI/forecasting_models.jl")
include("SI/sampling.jl")
include("SI/loss.jl")
include("SI/training.jl")
include("SI/time_stepping.jl")
include("SI/diffusion.jl")


# Models
export StochasticInterpolantModel
export ConditionalStochasticInterpolant
export FollmerStochasticInterpolant
export DataDependentCouplingStochasticInterpolant

# Sampling
export sde_sampler
export ode_sampler
export SDE_runge_kutta
export forecasting_sde_sampler
export forecasting_ode_sampler
export SDE_heun
export ODE_runge_kutta

# Interpolants
export Interpolant

# Loss
export get_loss
export get_forecasting_loss

# Training
export train_stochastic_interpolant

# Time stepping
export compute_multiple_SDE_steps
export compute_multiple_ODE_steps

# Diffusion
export Gamma
export DiffusionCoefficient



##### Utils #####
include("plotting_utils.jl")
include("checkpoint_utils.jl")
include("data_utils.jl")
include("preprocessing_utils.jl")
include("testing_utils.jl")

# Plotting
export create_gif

# Checkpoint
export save_checkpoint
export load_checkpoint

# Data
export load_transonic_cylinder_flow_data
export load_isotropic_turbulence_data

# Preprocessing
export StandardizeData
export NormalizePars

# Testing
export compute_RMSE
export compute_spatial_frequency
export compute_temporal_frequency


##### Training #####
include("training.jl")
export train_diffusion_model
export train_stochastic_interpolant

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
include("SMLD/loss.jl")
include("SMLD/sampling.jl")

# Loss
export get_loss

# Models
export ScoreMatchingLangevinDynamics

# Sampling
export euler_maruyama_sampler
export smld_sde_sampler
export smld_ode_sampler



end
