ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "20GiB"
ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"

using StochasticInterpolants
using Lux
using YAML
using Random
using NPZ
using LuxCUDA
using Optimisers
using FileIO
using Setfield

# For running on CPU.
# Consider reducing the sizes of DNS, LES, and CNN layers if
# you want to test run on a laptop.
T = Float32;
ArrayType = Array;
device = identity;
clean() = nothing;

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();

# Choose between "transonic_cylinder_flow", "incompressible_flow", "turbulence_in_periodic_box"
test_case = "transonic_cylinder_flow";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_interpolation";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
mask = mask |> dev;

num_train = size(trainset, 5);
num_steps = size(trainset, 4);
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3);



# Create a gif
# x1, p1 = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2), trainset_pars[1, 1, 1];
# x2, p2 = sqrt.(trainset[:, :, 1, :, 2].^2 + trainset[:, :, 2, :, 2].^2), trainset_pars[1, 1, 2];
# x3, p3 = sqrt.(trainset[:, :, 1, :, 3].^2 + trainset[:, :, 2, :, 3].^2), trainset_pars[1, 1, 3];
# x4, p4 = sqrt.(trainset[:, :, 1, :, 4].^2 + trainset[:, :, 2, :, 4].^2), trainset_pars[1, 1, 4];
# x5, p5 = sqrt.(trainset[:, :, 1, :, 5].^2 + trainset[:, :, 2, :, 5].^2), trainset_pars[1, 1, 5];
# x6, p6 = sqrt.(trainset[:, :, 1, :, 6].^2 + trainset[:, :, 2, :, 6].^2), trainset_pars[1, 1, 6];
# x7, p7 = sqrt.(trainset[:, :, 1, :, 7].^2 + trainset[:, :, 2, :, 7].^2), trainset_pars[1, 1, 7];
# x8, p8 = sqrt.(trainset[:, :, 1, :, 8].^2 + trainset[:, :, 2, :, 8].^2), trainset_pars[1, 1, 8];

# create_gif(
#     (x1, x2, x3, x4, x5, x6, x7, x8), 
#     "HF.gif", 
#     ("Ma = $p1", "Ma = $p2", "Ma = $p3", "Ma = $p4", "Ma = $p5", "Ma = $p6", "Ma = $p7", "Ma = $p8")
# )



##### Hyperparameters #####
# model_config = YAML.load_file("configs/neural_networks/$test_case.yml");

len_history = 2;

trainset_init_distribution, trainset_target_distribution, trainset_pars_distribution = prepare_data_for_time_stepping(
    trainset,
    trainset_pars;
    len_history=len_history
);

# Load autoencoder weights
if test_case == "transonic_cylinder_flow"
    num_latent_channels = 4;
    autoencoder = VariationalAutoencoder(C, (H, W), num_latent_channels, [32, 64, 128], "constant");
    ps_autoencoder, st_autoencoder, _ = load_checkpoint("transonic_VAE/best_model.bson") .|> dev;
elseif test_case == "incompressible_flow"
    autoencoder = VariationalAutoencoder(C, (H, W), num_latent_channels, [32, 64, 128], "constant");
    ps_autoencoder, st_autoencoder, _ = load_checkpoint("incompressible_VAE/best_model.bson") .|> dev;
elseif test_case == "turbulence_in_periodic_box"
    num_latent_channels = 4;
    autoencoder = VariationalAutoencoder(C, (H, W), num_latent_channels, [16, 32, 64, 128], "periodic");
    ps_autoencoder, st_autoencoder, _ = load_checkpoint("turbulence_VAE/best_model.bson") .|> dev;
end;
st_autoencoder = Lux.testmode(st_autoencoder);
autoencoder = VAE_wrapper(autoencoder, ps_autoencoder, st_autoencoder);


embedding_dims = 128;
batch_size = 8;
learning_rate = T(1e-4);
weight_decay = T(1e-8);
num_epochs = 4000;
channels = [32, 64];
attention_type = "standard"; # "linear" or "standard" or "DiT"
use_attention_in_layer = [false, false, false, false]; # [true, true, true, true];
attention_embedding_dims = 128;
num_heads = 4;
projection = nothing #project_onto_divergence_free;
padding = "periodic";

##### Forecasting SI model #####
# Define the velocity model
velocity = AttnParsConvNextUNet(
    (autoencoder.latent_dimensions[1], autoencoder.latent_dimensions[2]); 
    in_channels=autoencoder.latent_dimensions[3], 
    channels=channels, 
    embedding_dims=embedding_dims, 
    pars_dim=num_pars,
    len_history=len_history,
    attention_type=attention_type,
    use_attention_in_layer=use_attention_in_layer,
    padding=padding,
    attention_embedding_dims=attention_embedding_dims,
    num_heads=num_heads,
);
# velocity = ConditionalDiffusionTransformer(
#     (autoencoder.latent_dimensions[1], autoencoder.latent_dimensions[2]); 
#     in_channels=autoencoder.latent_dimensions[3], 
#     patch_size=(2, 2),
#     embedding_dims=256, 
#     depth=4, 
#     number_heads=8,
#     mlp_ratio=4.0f0, 
#     # dropout_rate=0.1f0, 
#     # embedding_dropout_rate=0.1f0,
#     pars_dim=num_pars,
#     len_history=len_history,
# );

# Define interpolant and diffusion coefficients
diffusion_multiplier = 0.01f0;

gamma = t -> diffusion_multiplier.* (1f0 .- t);
dgamma_dt = t -> -1f0 .* diffusion_multiplier; #ones(size(t)) .* diffusion_multiplier;
diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));

alpha = t -> 1f0 .- t; 
dalpha_dt = t -> -1f0;

beta = t -> t.^2;
dbeta_dt = t -> 2f0 .* t;

# Initialise the SI model
model = LatentFollmerStochasticInterpolant(
    velocity,
    autoencoder;
    interpolant=Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
    diffusion_coefficient=diffusion_coefficient,
    projection=projection,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

model_save_dir = "trained_models/latent_forecasting_model";

##### Load checkpoint #####
continue_training = false;
if continue_training
    if test_case == "transonic_cylinder_flow"
        best_model = "checkpoint_transonic_best"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "incompressible_flow"   
        best_model = "best_incompressible_model"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "turbulence_in_periodic_box"
        best_model = "turbulence_best_follmer_model"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    end;
end;
    
##### Train stochastic interpolant #####
ps, st = train_stochastic_interpolant(
    model=model,
    ps=ps,
    st=st,
    opt_state=opt_state,
    trainset_target_distribution=trainset_target_distribution,
    trainset_init_distribution=trainset_init_distribution,
    trainset_pars_distribution=trainset_pars_distribution,
    testset=testset,
    testset_pars=testset_pars,
    num_test_paths=4,
    model_save_dir=model_save_dir,
    num_epochs=num_epochs,
    batch_size=batch_size,
    normalize_data=normalize_data,
    mask=mask,
    rng=rng,
    dev=dev
);




##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
gif_save_path = "output/ode_SI";
num_test_paths = 5;

compare_ode_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    normalize_data,
    mask,
    50,
    dev,
    gif_save_path,
)

compare_sde_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    num_test_paths,
    normalize_data,
    mask,
    100,
    gif_save_path,
    rng,
    dev,
)













