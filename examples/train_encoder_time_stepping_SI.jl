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
continue_training = false;
model_base_dir = "trained_models/";
model_name = "encoder_forecasting_model";

if continue_training
    checkpoint_manager = CheckpointManager(
        test_case, model_name; base_folder=model_base_dir
    );

    config = checkpoint_manager.neural_network_config("trained_models/$test_case/$model_name");
else
    config = YAML.load_file("configs/neural_networks/$test_case.yml");
    
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
# velocity = DitParsConvNextUNet(
velocity = get_SI_neural_network(
    image_size=(H, W),
    model_params=config["model_args"],
)

encoder = get_encoder_neural_network(
    image_size=(H, W),
    model_params=config["model_args"],
)

# velocity = AttnParsConvNextUNet(
#     (H, W); 
#     in_channels=C, 
#     channels=config["model_args"]["channels"], 
#     embedding_dims=config["model_args"]["embedding_dims"], 
#     pars_dim=num_pars,
#     len_history=config["model_args"]["len_history"],
#     attention_type=config["model_args"]["attention_type"],
#     use_attention_in_layer=config["model_args"]["use_attention_in_layer"],
#     padding=config["model_args"]["padding"],
#     attention_embedding_dims=config["model_args"]["attention_embedding_dims"],
#     num_heads=config["model_args"]["num_heads"],
# );
# encoder = velocity;

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

# Initialise the SI model
model = EncoderFollmerStochasticInterpolant(
    velocity, encoder, interpolant;
    diffusion_coefficient=diffusion_coefficient,
    projection=config["model_args"]["projection"],
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
    ps, st, opt_state = weights_and_states .|> dev;
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

##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
gif_save_path = "output/ode_SI";
num_test_paths = 5;

compare_sde_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    num_test_paths,
    normalize_data,
    mask,
    150,
    gif_save_path,
    rng,
    dev,
)

CUDA.reclaim()
GC.gc()

