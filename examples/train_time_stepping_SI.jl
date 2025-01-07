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
test_case = "kolmogorov";

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
continue_training = true;
model_base_dir = "trained_models/";
model_name = "forecasting_model_not_optimized";

if continue_training
    checkpoint_manager = CheckpointManager(
        test_case, model_name; base_folder=model_base_dir
    )

    config = checkpoint_manager.neural_network_config
else
    # config = YAML.load_file("configs/neural_networks/$(test_case)_dit.yml");
    config = YAML.load_file("configs/neural_networks/$test_case.yml")
    
    checkpoint_manager = CheckpointManager(
        test_case, model_name; 
        neural_network_config=config, 
        data_config=YAML.load_file("configs/test_cases/$test_case.yml"),
        base_folder=model_base_dir
    )
end;

trainset = prepare_data_for_time_stepping(  
    trainset,
    trainset_pars;
    len_history=config["model_args"]["len_history"]
);

##### Forecasting SI model #####
# Define the velocity model
velocity = get_SI_neural_network(;
    image_size=(H, W),
    model_params=config["model_args"]
);


# Get Interpolant
interpolant = get_interpolant(
    config["interpolant_args"]["alpha"],
    config["interpolant_args"]["beta"],
    config["interpolant_args"]["gamma"],
    T(config["interpolant_args"]["gamma_multiplier"]),
);

# coefs = [
#     -1.49748  -0.181875   0.127546  -0.0188962  -0.0104655  -0.00495827  -0.00481044  -0.00316509;
#     -1.89775   0.182452  -0.224247   0.0196697  -0.0101798   0.00545455  -0.00265759   0.00324879;
#     -4.94467  -1.35222   -0.68816   -0.400287   -0.276452   -0.174084    -0.11204     -0.0665296
# ];
# # Cast to float32
# coefs = coefs .|> T;

# coefs = [
#     -1.05734  -0.00348673  -0.0312818  -0.00382112  -0.00580364;
#     -1.05611   0.00127347  -0.0293777   0.00343358  -0.00645624
# ]
# coefs = coefs .|> T;

# interpolant = Interpolant(
#     t -> get_alpha_series(t, coefs[1, :]), 
#     t -> get_beta_series(t, coefs[2, :]), 
#     t -> get_dalpha_series_dt(t, coefs[1, :]),
#     t -> get_dbeta_series_dt(t, coefs[2, :]), 
#     t -> 0.1f0 .* (1f0 .- t),
#     t -> -1f0 .* 0.1f0
# )

# Get diffusion coefficient
diffusion_coefficient = get_diffusion_coefficient(
    config["diffusion_args"]["type"],
    T(config["diffusion_args"]["multiplier"]),
);

if config["model_args"]["projection"] == "divergence_free"
    projection = project_onto_divergence_free;
else
    projection = nothing;
end;

# Initialise the SI model
model = FollmerStochasticInterpolant(
    velocity; 
    # interpolant=interpolant, #Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
    interpolant=interpolant,
    diffusion_coefficient=diffusion_coefficient, #t -> get_gamma_series(t, coefs[3, :]), #diffusion_coefficient,
    projection=projection,
    len_history=config["model_args"]["len_history"],
    dev=dev
);
##### Optimizer #####
opt = Optimisers.AdamW(
    T(config["optimizer_args"]["learning_rate"]), 
    (0.9f0, 0.99f0), 
    T(config["optimizer_args"]["weight_decay"])
);

##### Load model #####
if continue_training
    weights_and_states = checkpoint_manager.load_model();
    ps = weights_and_states.ps |> dev;
    st = weights_and_states.st |> dev;
    opt_state = Optimisers.setup(opt, ps);
else
    ps, st = Lux.setup(rng, model) .|> dev;
    opt_state = Optimisers.setup(opt, ps);
end;
    
##### Train stochastic interpolant #####
ps, st = train_stochastic_interpolant(
    model=model,
    ps=ps,
    st=st,
    opt_state=opt_state,
    trainset=trainset, 
    testset=(state=testset, pars=testset_pars),
    checkpoint_manager=checkpoint_manager,
    training_args=config["training_args"],
    normalize_data=normalize_data,
    mask=mask,  
    rng=rng,
    dev=dev
);


CUDA.reclaim()
GC.gc()































# Create a gif
x1, p1 = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2), trainset_pars[1, 1, 1];
x2, p2 = sqrt.(trainset[:, :, 1, :, 2].^2 + trainset[:, :, 2, :, 2].^2), trainset_pars[1, 1, 2];
x3, p3 = sqrt.(trainset[:, :, 1, :, 3].^2 + trainset[:, :, 2, :, 3].^2), trainset_pars[1, 1, 3];
x4, p4 = sqrt.(trainset[:, :, 1, :, 4].^2 + trainset[:, :, 2, :, 4].^2), trainset_pars[1, 1, 4];
x5, p5 = sqrt.(trainset[:, :, 1, :, 5].^2 + trainset[:, :, 2, :, 5].^2), trainset_pars[1, 1, 5];
x6, p6 = sqrt.(trainset[:, :, 1, :, 6].^2 + trainset[:, :, 2, :, 6].^2), trainset_pars[1, 1, 6];
x7, p7 = sqrt.(trainset[:, :, 1, :, 7].^2 + trainset[:, :, 2, :, 7].^2), trainset_pars[1, 1, 7];
x8, p8 = sqrt.(trainset[:, :, 1, :, 8].^2 + trainset[:, :, 2, :, 8].^2), trainset_pars[1, 1, 8];

create_gif(
    (x1, x2, x3, x4, x5, x6, x7, x8), 
    "HF.gif", 
    ("Ma = $p1", "Ma = $p2", "Ma = $p3", "Ma = $p4", "Ma = $p5", "Ma = $p6", "Ma = $p7", "Ma = $p8")
)


















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


##### Hyperparameters #####
# model_config = YAML.load_file("configs/neural_networks/$test_case.yml");

len_history = 2;

trainset_init_distribution, trainset_target_distribution, trainset_pars_distribution = prepare_data_for_time_stepping(
    trainset,
    trainset_pars;
    len_history=len_history
);

embedding_dims = 256;
batch_size = 4;
learning_rate = T(1e-4);
weight_decay = T(1e-8);
num_epochs = 500;
channels = [16, 32, 64, 128];
if test_case == "transonic_cylinder_flow"
    attention_type = "linear"; # "linear" or "standard" or "DiT"
    use_attention_in_layer = [true, true, true, true]
    attention_embedding_dims = 32;
    num_heads = 4;
    padding = "constant";    
elseif test_case == "turbulence_in_periodic_box"
    attention_type = "DiT"; # "linear" or "standard" or "DiT"
    use_attention_in_layer = [false, false, false, false];
    attention_embedding_dims = 256
    num_heads = 8;
    padding = "periodic";
end;
projection = nothing #project_onto_divergence_free;

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

gamma = t -> diffusion_multiplier.* (1f0 .- t);
dgamma_dt = t -> -1f0 .* diffusion_multiplier; #ones(size(t)) .* diffusion_multiplier;
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
    projection=projection,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

model_save_dir =  "trained_models/$test_case";

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);
    
continue_training = true;
if continue_training
    if test_case == "transonic_cylinder_flow"
        best_model = "checkpoint_epoch_18"
        ps, st, opt_state = load_checkpoint("trained_models/transonic_cylinder_flow/$best_model.bson") .|> dev;
    elseif test_case == "incompressible_flow"
        best_model = "best_incompressible_model"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "turbulence_in_periodic_box"
        best_model = "best_model"
        ps, st, opt_state = load_checkpoint("trained_models/turbulence_in_periodic_box/$best_model.bson") .|> dev;
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



save_checkpoint(
    ps=ps, 
    st=st,
    opt_st=opt_state,
    output_dir=model_save_dir,
    epoch=10
)














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
    25,
    gif_save_path,
    rng,
    dev,
)






