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
# options are "pars_low", "pars_high" for "incompressible_flow" test case
test_args = "pars_low";

# Load the test case configuration
test_case_config = YAML.load_file("configs/test_cases/$test_case.yml");

# Get data path
data_folder = test_case_config["data_folder"];

# Get dimensions of state and parameter spaces
H = test_case_config["state_dimensions"]["height"];
W = test_case_config["state_dimensions"]["width"];
C = test_case_config["state_dimensions"]["channels"];
if !isnothing(test_case_config["parameter_dimensions"])
    pars_dim = length(test_case_config["parameter_dimensions"]);
    num_pars = pars_dim
else
    pars_dim = 1;
    num_pars = 0;
end;

# Load mask if it exists
if test_case_config["with_mask"]
    mask = npzread("$data_folder/sim_000000/obstacle_mask.npz")["arr_0"];
    mask = permutedims(mask, (2, 1)) |> dev;
else
    mask = ones(H, W, C) |> dev;
end;


# Time step information
test_start_time = test_case_config["test_args"][test_args]["time_step_info"]["start_time"];
test_num_steps = test_case_config["test_args"][test_args]["time_step_info"]["num_steps"];
test_skip_steps = test_case_config["test_args"][test_args]["time_step_info"]["skip_steps"];


if test_case == "transonic_cylinder_flow"

    # Load the test data
    testset, testset_pars = load_transonic_cylinder_flow_data(
        data_folder=data_folder,
        data_ids=test_case_config["test_args"][test_args]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(test_start_time, test_num_steps, test_skip_steps)
    );
elseif test_case == "incompressible_flow"

    # Load the test data
    testset, testset_pars = load_incompressible_flow_data(
        data_folder=data_folder,
        data_ids=test_case_config["test_args"][test_args]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(test_start_time, test_num_steps, test_skip_steps)
    );

elseif test_case == "turbulence_in_periodic_box"
    # Load the test data
    testset, testset_pars = load_turbulence_in_periodic_box_data(
        data_folder=data_folder,
        data_ids=test_case_config["test_args"][test_args]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(test_start_time, test_num_steps, test_skip_steps)
    );
else
    error("Invalid test case")
end;


if test_case_config["normalize_data"]
    # Normalize the data
    normalize_data = StandardizeData(
        test_case_config["norm_mean"], 
        test_case_config["norm_std"],
    );
    trainset = normalize_data.transform(trainset);
    testset = normalize_data.transform(testset);

    # Normalize the parameters
    normalize_pars = NormalizePars(
        test_case_config["pars_min"], 
        test_case_config["pars_max"]
    );
    trainset_pars = normalize_pars.transform(trainset_pars);
    testset_pars = normalize_pars.transform(testset_pars);
else
    normalize_data = nothing;
end;

# Apply mask
mask = mask |> cpu_dev;
trainset = trainset .* mask;
testset = testset .* mask;


##### Hyperparameters #####
len_history = 2;
embedding_dims = 128;
batch_size = 8;
learning_rate = T(1e-4);
weight_decay = T(1e-8);
num_epochs = 4000;
channels = [8, 16, 32, 64];
attention_type = "DiT"; # "linear" or "standard" or "DiT"
use_attention_in_layer = [false, false, false, false]; # [true, true, true, true];
attention_embedding_dims = 64;
num_heads = 4;
projection = nothing 

##### Forecasting SI model #####
# Define the velocity model
velocity = AttnParsConvNextUNet(
# velocity = DitParsConvNextUNet(
    (H, W); 
    in_channels=C, 
    channels=channels, 
    embedding_dims=embedding_dims, 
    pars_dim=num_pars,
    len_history=len_history,
    use_attention_in_layer=use_attention_in_layer,
    padding="periodic"
);

# Define interpolant and diffusion coefficients
diffusion_multiplier = 1.0f0;

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
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

##### Load checkpoint #####
continue_training = true;
if continue_training
    if test_case == "transonic_cylinder_flow"
        best_model = "checkpoint_transonic_best"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "incompressible_flow"
        best_model = "best_incompressible_model"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "turbulence_in_periodic_box"
        best_model = "checkpoint_epoch_300"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    end;
end;


##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
num_test_paths = 5;


for num_generator_steps = [75];
    print("Number of generator steps: ", num_generator_steps)

    gif_save_path = "output/num_generator_steps_$num_generator_steps";

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

