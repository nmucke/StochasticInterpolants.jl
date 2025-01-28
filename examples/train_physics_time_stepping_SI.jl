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
using IncompressibleNavierStokes
using Zygote

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
# model_config = YAML.load_file("configs/neural_networks/$test_case.yml");

len_history = 2;

trainset_init_distribution, trainset_target_distribution, trainset_pars_distribution = prepare_data_for_time_stepping(
    trainset,
    trainset_pars;
    len_history=len_history
);



embedding_dims = 128;
batch_size = 8;
learning_rate = T(1e-4);
weight_decay = T(1e-8);
num_epochs = 500;
channels = [8, 16, 32, 64];
if test_case == "transonic_cylinder_flow"
    attention_type = "linear"; # "linear" or "standard" or "DiT"
    use_attention_in_layer = [false, false, false, false]
    attention_embedding_dims = 32;
    num_heads = 4;
    padding = "constant";    
elseif test_case == "turbulence_in_periodic_box"
    attention_type = "standard"; # "linear" or "standard" or "DiT"
    use_attention_in_layer = [false, false, false, false];
    attention_embedding_dims = 32
    num_heads = 4;
    padding = "periodic";
end;
projection = nothing #project_onto_divergence_free;

##### Forecasting SI model #####
# Define the velocity model
# velocity = DitParsConvNextUNet(
model_velocity = AttnParsConvNextUNet(
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

# Setup
Re = 10000f0;
n = 128
physics_dt = 5f-5 * 10f0 * 4f0;
ax = LinRange(0f0, 1f0, n + 1);
setup = Setup(; x = (ax, ax), Re, ArrayType = CuArray{T});
# psolver = psolver_spectral(setup);


create_right_hand_side(setup, psolver) = function right_hand_side(u)
    (; Iu) = setup.grid
    u = pad_circular(u, 1; dims = 1:2)
    out = Array{eltype(u)}[]
    for i in 1:size(u, 4)
        t = zero(eltype(u))
        u_ = u[:, :, 1, i], u[:, :, 2, i]
        F = IncompressibleNavierStokes.momentum(u_, nothing, t, setup)
        F = cat(F[1], F[2]; dims = 3)
        F = F[2:end-1, 2:end-1, :]
        out = [out; [F]]
    end
    stack(out; dims = 4)
end

f = create_right_hand_side(setup, nothing);
physics_velocity(input) = begin
    x, x_0, pars, t = input
    return (5f-5 * 10f0 * 4f0) * f(x_0[:, :, :, end, :]);
end;

# x = randn(rng, (H, W, C, len_history, batch_size)) .|> T |> dev;
# out = physics_velocity((x, x, x, x))
# gradient(x -> sum(physics_velocity((x, x, x, x))), x[:, :, :, :, 1])


# Define interpolant and diffusion coefficients
diffusion_multiplier = 0.1f0;

gamma = t -> diffusion_multiplier.* (1f0 .- t);
dgamma_dt = t -> -1f0 .* diffusion_multiplier; #ones(size(t)) .* diffusion_multiplier;
diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));

alpha = t -> 1f0 .- t; 
dalpha_dt = t -> -1f0;

beta = t -> t;
dbeta_dt = t -> 1f0;

# Initialise the SI model
model = PhysicsInformedStochasticInterpolant(
    model_velocity, physics_velocity; 
    interpolant=Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
    diffusion_coefficient=diffusion_coefficient,
    projection=projection,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

model_save_dir = "trained_models/physics_models/$test_case";

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Load checkpoint #####
continue_training = false;
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

