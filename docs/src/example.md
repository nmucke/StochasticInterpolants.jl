```@meta
CurrentModule = StochasticInterpolants
```

# Example


```julia
# Load relevant packages
using StochasticInterpolants
using Random
using Lux
using LuxCUDA
using Optimisers

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();

# Choose test problem
test_case = "kolmogorov";

# Choose test case
test_args = "pars_low";
```

The test problem and test case determines which data and model config file to load. The data config file contains information about the data, such as the mask, the normalizer, and the number of parameters. The model config file contains information about the model and the training, such as the neural network architecture, the optimizer, and the training parameters.

```julia
# Load train data, test data, data normalizer, and mask given by the config file
# Depending on the test case, mask and normalize_data might be nothing
trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
mask = mask |> dev;

num_train = size(trainset, 5); # Number of training trajectories
num_steps = size(trainset, 4); # Number of training steps
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3); # Dimensions of the data
```

```julia
# Setup checkpoint manager that handles saving and loading of models
# and the training progress
model_base_dir = "trained_models/";
model_name = "forecasting_model_not_optimized";

config = YAML.load_file("configs/neural_networks/$test_case.yml");

checkpoint_manager = CheckpointManager(
    test_case, model_name; 
    neural_network_config=config, 
    data_config=YAML.load_file("configs/test_cases/$test_case.yml"),
    base_folder=model_base_dir
);
```


```julia
# Prepare the data for time stepping
# This function takes the training data and the training parameters
# and returns the data in the format required for time stepping
trainset = prepare_data_for_time_stepping(  
    trainset,
    trainset_pars;
    len_history=config["model_args"]["len_history"]
);
```

```julia
# Define the velocity model
velocity = get_SI_neural_network(;
    image_size=(H, W),
    model_params=config["model_args"]
);
```

```julia
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

# Get projection if specified in the config
if config["model_args"]["projection"] == "divergence_free"
    projection = project_onto_divergence_free;
else
    projection = nothing;
end;
```


```julia
# Setup the SI model
model = FollmerStochasticInterpolant(
    velocity; 
    interpolant=interpolant,
    diffusion_coefficient=diffusion_coefficient, #t -> get_gamma_series(t, coefs[3, :]), #diffusion_coefficient,
    projection=projection,
    len_history=config["model_args"]["len_history"],
    dev=dev
);

# Initialise model and move to device
ps, st = Lux.setup(rng, model) .|> dev;

# Setup the optimiser
opt = Optimisers.AdamW(
    T(config["optimizer_args"]["learning_rate"]), 
    (0.9f0, 0.99f0), 
    T(config["optimizer_args"]["weight_decay"])
);

# Initialise the optimiser
opt_state = Optimisers.setup(opt, ps);
```

```julia
# Train the model
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
```