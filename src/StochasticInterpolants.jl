"""
    StochasticInterpolants

Implementation of the Stochastic Interpolants method for generative modeling.
"""
module StochasticInterpolants

# using Lux
# using Random
# using CUDA
# using NNlib
# using Setfield
# using LuxCUDA
# using Boltz
using Infiltrator

##### Layers #####
include("neural_network_layers/conv_next.jl")
include("neural_network_layers/embeddings.jl")
include("neural_network_layers/transformer.jl")
# include("neural_network_layers/conv.jl")
include("neural_network_layers/autoencoder.jl")
include("neural_network_layers/diffusion_transformer.jl")
include("neural_network_layers/neural_network_utils.jl")
include("neural_network_layers/full_networks.jl")

export parameter_diffusion_transformer_block

export pars_cat
export transform_to_nothing
export Identity
export StateParsIdentity
export get_padding
export get_attention_layer
export get_t_pars_embedding

export transform_to_nothing
export LinearMultiHeadSelfAttention
export LinearSpatialAttention
export MultiHeadSelfAttention
export residual_block
export conv_next_block
export conv_next_block_no_pars
export get_history_state_embedding
export ConvNextUNetNoPars
export ConvNextUNetWithPars
export multiple_conv_next_blocks
export sinusoidal_embedding
export modulate
export patchify
export unpatchify
export reshape_modulation
export DiffusionTransformerBlock
export FinalLayer
export DiffusionTransformer
export ConditionalDiffusionTransformer
export ConvNextUNet
export parameter_diffusion_transformer_block
export SpatialAttention
export DitParsConvNextUNet
export AttnParsConvNextUNet
export conv_next_block_no_pars

export MultipleBlocks
export DownBlock
export UpBlock
export Encoder
export VariationalEncoder
export Decoder
export Autoencoder
export VariationalAutoencoder
export VAE_wrapper
export get_SI_neural_network
export get_encoder_neural_network


include("unet_transformer.jl")
export dit_down_block
export dit_up_block


##### Utils #####
include("plotting_utils.jl")
include("checkpoint_utils.jl")
include("data_utils.jl")
include("preprocessing_utils.jl")
include("testing_utils.jl")
include("projections.jl")

# Plotting
export create_gif

# Checkpoint
export save_checkpoint
export load_checkpoint
export CheckpointManager
export load_model_weights

# Data
export load_transonic_cylinder_flow_data
export load_incompressible_flow_data
export load_turbulence_in_periodic_box_data
export load_kolmogorov_data
export prepare_data_for_time_stepping
export prepare_latent_data
export load_test_case_data

# Preprocessing
export StandardizeData
export NormalizePars

# Testing
export compute_RMSE
export compute_energy_spectra
export compute_spatial_frequency
export compute_temporal_frequency
export compare_sde_pred_with_true
export compare_ode_pred_with_true
export compute_total_energy
export compute_inner_product
export compute_norm

# Projections
export divfunc
export gradfunc
export laplacefunc
export project_onto_divergence_free


##### SI #####
include("SI/diffusion.jl")
include("SI/interpolants.jl")
include("SI/models.jl")
# include("SI/conditional_models.jl")
include("SI/forecasting_models.jl")
include("SI/physics_informed_models.jl")
include("SI/sampling.jl")
include("SI/loss.jl")
include("SI/training.jl")
include("SI/time_stepping.jl")


# Models
export StochasticInterpolantModel
# export ConditionalStochasticInterpolant
export FollmerStochasticInterpolant
export LatentFollmerStochasticInterpolant
export DataDependentCouplingStochasticInterpolant
export PhysicsInformedStochasticInterpolant
export EncoderFollmerStochasticInterpolant

# Sampling
export sde_sampler
export ode_sampler
export SDE_runge_kutta
export forecasting_sde_sampler
export forecasting_latent_sde_sampler
export forecasting_ode_sampler
export SDE_heun
export ODE_runge_kutta

# Diffusion
# export DiffusionCoefficient
export get_diffusion_coefficient

# Interpolants
export Interpolant
export get_interpolant
export get_alpha_series
export get_beta_series
export get_gamma_series
export get_dalpha_series_dt
export get_dbeta_series_dt
export get_dgamma_series_dt
export d_interpolant_energy_dt
export compute_physics_consistent_interpolant_coefficients
export interpolant_velocity
export objective_function


# Loss
export get_loss
export get_forecasting_loss
export get_forecasting_from_gaussian_loss
export get_physics_forecasting_loss
export get_encoder_forecasting_loss

# Training
export train_stochastic_interpolant
export train_stochastic_interpolant_for_closure

# Time stepping
export compute_multiple_SDE_steps
export compute_multiple_latent_SDE_steps
export compute_multiple_ODE_steps



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
