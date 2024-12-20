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
using BSON
using Serialization

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
test_case = "iris";

len_history = 1;
num_train_trajectories = 28;
num_test_trajectories = 2;

H = 64;
W = 64;
C = 2;
num_steps = 3000;
skip_steps = 5;


v_train = Serialization.deserialize("data/iris/v_train_data_FVM.bson") |> cpu_dev;
c_train = Serialization.deserialize("data/iris/c_train_data_FVM.bson") |> cpu_dev;
trainset_pars = zeros(1, num_steps, num_train_trajectories);

v_train = v_train[:, :, :, 1:skip_steps:num_steps, :];
c_train = c_train[:, :, :, 1:skip_steps:num_steps, :];
trainset_pars = trainset_pars[:, 1:skip_steps:num_steps, :];
    
init_distribution = reshape(v_train, H, W, C, len_history, Int(num_steps/skip_steps*num_train_trajectories));
target_distribution = reshape(c_train, H, W, C, Int(num_steps/skip_steps*num_train_trajectories));
pars_distribution = reshape(trainset_pars, 1, Int(num_steps/skip_steps*num_train_trajectories));

trainset = (;
    init_distribution=init_distribution,
    target_distribution=target_distribution,
    pars_distribution=pars_distribution
);


v_test = Serialization.deserialize("data/iris/v_test_data_FVM.bson") |> cpu_dev;
c_test = Serialization.deserialize("data/iris/c_test_data_FVM.bson") |> cpu_dev;
pars_test = zeros(1, num_steps, num_test_trajectories);

v_test = v_test[:, :, :, 1:skip_steps:num_steps, 1:num_test_trajectories];
c_test = c_test[:, :, :, 1:skip_steps:num_steps, 1:num_test_trajectories];
pars_test = pars_test[:, 1:skip_steps:num_steps, 1:num_test_trajectories];
    
init_distribution = reshape(v_test, H, W, C, len_history, Int(num_steps/skip_steps*num_test_trajectories));
target_distribution = reshape(c_test, H, W, C, Int(num_steps/skip_steps*num_test_trajectories));
pars = reshape(pars_test, 1, Int(num_steps/skip_steps*num_test_trajectories));

testset = (;
    init_distribution=init_distribution,
    target_distribution=target_distribution,
    pars=pars
);


##### Hyperparameters #####
continue_training = false;
model_base_dir = "trained_models/";
model_name = "closure_model";

config = YAML.load_file("configs/neural_networks/$test_case.yml");
checkpoint_manager = CheckpointManager(
    test_case, model_name; 
    neural_network_config=config, 
    data_config=YAML.load_file("configs/test_cases/$test_case.yml"),
    base_folder=model_base_dir
);


mask = ones(H, W, C) |> dev;


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
    interpolant=interpolant, #Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
    diffusion_coefficient=diffusion_coefficient,
    projection=projection,
    dev=dev,
    gaussian_base_distribution=false
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
    ps, st, opt_state = weights_and_states .|> dev;
else
    ps, st = Lux.setup(rng, model) .|> dev;
    opt_state = Optimisers.setup(opt, ps);
end;
    
##### Train stochastic interpolant #####
ps, st = train_stochastic_interpolant_for_closure(
    model=model,
    ps=ps,
    st=st,
    opt_state=opt_state,
    trainset=trainset, 
    testset=testset,
    checkpoint_manager=checkpoint_manager,
    training_args=config["training_args"],
    normalize_data=nothing,
    mask=mask,  
    rng=rng,
    dev=dev
);


CUDA.reclaim()
GC.gc()

