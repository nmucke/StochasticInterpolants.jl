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

CUDA.reclaim()
GC.gc()

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();


# Choose between "transonic_cylinder_flow", "incompressible_flow", "turbulence_in_periodic_box"
test_case = "turbulence_in_periodic_box";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
mask = mask |> dev;

num_train = size(trainset, 5);
num_steps = size(trainset, 4);
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3);


##### Hyperparameters #####
len_history = 2;
embedding_dims = 256;
batch_size = 8;
num_epochs = 1000;
channels = [16, 32, 64, 128];
projection = nothing #project_onto_divergence_free;
if test_case == "transonic_cylinder_flow"
    attention_type = "linear"; # "linear" or "standard" or "DiT"
    use_attention_in_layer = [true, true, true, true]
    attention_embedding_dims = 32;
    padding = "constant";
    num_heads = 4;
elseif test_case == "turbulence_in_periodic_box"
    attention_type = "DiT"; # "linear" or "standard" or "DiT"
    use_attention_in_layer = [false, false, false, false];
    attention_embedding_dims = 256
    padding = "periodic";
    num_heads = 8;
end;

##### Forecasting SI model #####
# Define the velocity model
# velocity = DitParsConvNextUNet(
velocity = AttnParsConvNextUNet(
    (H, W); 
    in_channels=C, 
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


# Define interpolant and diffusion coefficients
diffusion_multiplier = 0.5f0;
interpolant_multiplier = 0.5f0;


gamma = t -> interpolant_multiplier.* (1f0 .- t);
dgamma_dt = t -> -1f0 .* interpolant_multiplier; #ones(size(t)) .* diffusion_multiplier;
diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));

alpha = t -> 1f0 .- t; 
dalpha_dt = t -> -1f0;

beta = t -> t.^2;
dbeta_dt = t -> 2f0 .* t;

# Initialise the SI model
model = FollmerStochasticInterpolant(
    velocity; 
    interpolant=Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
    diffusion_coefficient=diffusion_coefficient,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

##### Load checkpoint #####
if test_case == "transonic_cylinder_flow"
    best_model = "checkpoint_epoch_10"
    ps, st, opt_state = load_checkpoint("trained_models/transonic_cylinder_flow/$best_model.bson") .|> dev;
elseif test_case == "incompressible_flow"
    best_model = "best_incompressible_model"
    ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
elseif test_case == "turbulence_in_periodic_box"
    # best_model = "best_model"
    best_model = "checkpoint_epoch_400"
    ps, st, opt_state = load_checkpoint("trained_models/turbulence_in_periodic_box/$best_model.bson") .|> dev;
end;


##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
num_test_paths = 4;


for num_generator_steps = [100];
    print("Number of generator steps: ", num_generator_steps)

    gif_save_path = "output";

    print("\n") 
    compare_sde_pred_with_true(
        model,
        ps,
        st_,
        testset,
        testset_pars,
        num_test_paths,
        normalize_data,
        mask,
        num_generator_steps,
        gif_save_path,
        rng,
        dev,
    )
    print("####################################################")
    print("\n")
end;

