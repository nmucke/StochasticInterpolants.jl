```@meta
CurrentModule = StochasticInterpolants
```

# Example

In this example a stochastic interpolant is trained to forecast Kolmogorov flow using the stochastic interpolant.

The example requires the following:
+ The data for the test case.
+ The model configuration file.
+ The data configuration file.


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

The model configuration file is loaded and the checkpoint manager is set up. The checkpoint manager handles saving and loading of models and the training progress. 

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

The data is prepared for time stepping according to the model configuration. Specifically, the data is reshaped to have an initial distribution and target distribution. The initial distribution is the data at time `t-len_history` to `t` and the target distribution is the data at time `t+1`. 

```julia
# Prepare the data for time stepping according to model configuration
# This function takes the training data and the training parameters
# and returns the data in the format required for time stepping
trainset = prepare_data_for_time_stepping(  
    trainset,
    trainset_pars;
    len_history=config["model_args"]["len_history"]
);
```

The velocity model is neural network. The architecture can be anything that suits the data dimensions. In general, the architecture is similar to architectures used in denosing diffusion models. In this project, the architecture is a UNet with ConvNext layers and diffusion transformer in the bottleneck. The hyperparameters are set in config file.  
```julia
# Define the velocity model
velocity = get_SI_neural_network(;
    image_size=(H, W),
    model_params=config["model_args"]
);
```

The interpolant and diffusion coefficient are set according to the config file. The interpolant is a stochastic interpolant with the following form:
```math
\begin{equation}
    I_t = \alpha(t)X_0 + \beta(t)X_1 + \gamma(t)W_t
\end{equation}
```
In this example 'alpha', 'beta', and 'gamma' are defined as follows:
```math
\begin{equation}
    \alpha(t) = 1 - t, \quad \beta(t) = t^2, \quad \gamma(t) = 0.1 * (1 - t).
\end{equation}
With this formulation, the resulting SDE is of the form:
```math
\begin{equation}
    dX_t = \left[b(X_t, X_0, t) + (g^2(t) - \gamma^2(t)) \right] dt + g(t)dW_t,
\end{equation}
```
where `g` can be chosen (almost) freely. 

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
    config["optimizer_args"]["learning_rate"], 
    (0.9f0, 0.99f0), 
    config["optimizer_args"]["weight_decay"]
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

For easy testing of the model, we can compare the model predictions with the true data. The function `compare_sde_pred_with_true` takes the model, the model weights, the model state, the test data, the test parameters, the number of paths to compute in the SDE ensemble, the data normalizer, the mask, the number of pseudo-steps in the SDE ensemble, the directory to save figures, the random number generator, and the device. The function returns the pathwise mean squared error and the mean mean squared error. The function saves the following figures in the directory specified:
+ Animation of the true data and the model predictions
+ Plot of the total energy evolution
+ Plot of energy spectrum at the last time step
+ plot of temporal energy spectrum and the center point
```julia
pathwise_MSE, mean_MSE = compare_sde_pred_with_true(
    model, # SI model
    ps, # Model weights
    Lux.testmode(st), # Model state in test mode
    testset.state, # Test data
    testset.pars, # Test parameters
    5, # Number of paths to compute in SDE ensemble
    normalize_data, # Data normalizer
    mask, # Mask
    50, # Number of pseudo-steps in SDE ensemble
    "$(checkpoint_manager.figures_dir)/forecasting", # Directory to save figures
    rng, # Random number generator
    dev, # Device
)
```