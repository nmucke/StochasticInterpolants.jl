ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "20GiB"
ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"


using StochasticInterpolants
using Lux
using Random
using CUDA
using NNlib
using Setfield
using MLDatasets
using Plots
using Optimisers
using Zygote
using LuxCUDA
using Images
using ImageFiltering
using Printf
using Statistics
using LinearAlgebra
using NPZ
using JSON
using YAML

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();

# Choose between "transonic_cylinder_flow" and "incompressible_cylinder_flow"
test_case = "transonic_cylinder_flow";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts"
test_args = "long_rollouts";

# Load the test case configuration
test_case_config = YAML.load_file("configs/test_cases/$test_case.yml");

# Get data path
data_folder = test_case_config["data_folder"];

# Get dimensions of state and parameter spaces
H = test_case_config["state_dimensions"]["height"];
W = test_case_config["state_dimensions"]["width"];
C = test_case_config["state_dimensions"]["channels"];
num_pars = length(test_case_config["parameter_dimensions"]);

# Load mask if it exists
if test_case_config["with_mask"]
    mask = npzread("$data_folder/sim_000000/obstacle_mask.npz")["arr_0"];
    mask = permutedims(mask, (2, 1)) |> dev;
else
    mask = ones(H, W, C) |> dev;
end;


# Number of training samples
num_train = length(test_case_config["training_args"]["ids"]);

# Time step information
start_time = test_case_config["training_args"]["time_step_info"]["start_time"];
num_steps = test_case_config["training_args"]["time_step_info"]["num_steps"];
skip_steps = test_case_config["training_args"]["time_step_info"]["skip_steps"];

test_start_time = test_case_config["test_args"][test_args]["time_step_info"]["start_time"];
test_num_steps = test_case_config["test_args"][test_args]["time_step_info"]["num_steps"];
test_skip_steps = test_case_config["test_args"][test_args]["time_step_info"]["skip_steps"];

# Load the training data
trainset, trainset_pars = load_transonic_cylinder_flow_data(
    data_folder=data_folder,
    data_ids=test_case_config["training_args"]["ids"],
    state_dims=(H, W, C),
    num_pars=num_pars,
    time_step_info=(start_time, num_steps, skip_steps)
);

# Load the test data
testset, testset_pars = load_transonic_cylinder_flow_data(
    data_folder=data_folder,
    data_ids=test_case_config["test_args"][test_args]["ids"],
    state_dims=(H, W, C),
    num_pars=num_pars,
    time_step_info=(test_start_time, test_num_steps, test_skip_steps)
);

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

# Apply mask
mask = mask |> cpu_dev;
trainset = trainset .* mask;
testset = testset .* mask;

# # Create a gif
# x1, p1 = trainset[:, :, 4, :, 1], trainset_pars[1, 1, 1];
# x2, p2 = trainset[:, :, 4, :, 2], trainset_pars[1, 1, 2];
# x3, p3 = trainset[:, :, 4, :, 3], trainset_pars[1, 1, 3];
# x4, p4 = trainset[:, :, 4, :, 4], trainset_pars[1, 1, 4];
# x5, p5 = trainset[:, :, 4, :, 5], trainset_pars[1, 1, 5];
# x6, p6 = trainset[:, :, 4, :, 6], trainset_pars[1, 1, 6];
# x7, p7 = trainset[:, :, 4, :, 7], trainset_pars[1, 1, 7];
# x8, p8 = trainset[:, :, 4, :, 8], trainset_pars[1, 1, 8];

# create_gif(
#     (x1, x2, x3, x4, x5, x6, x7, x8), 
#     "HF.gif", 
#     ("Ma = $p1", "Ma = $p2", "Ma = $p3", "Ma = $p4", "Ma = $p5", "Ma = $p6", "Ma = $p7", "Ma = $p8")
# )

# create_gif((x1, x2, x3), "HF.gif", ("Ma = $p1", "Ma = $p2", "Ma = $p3"))


# Divide the training set into initial and target distributions
trainset_init_distribution = trainset[:, :, :, 1:end-1, :];
trainset_target_distribution = trainset[:, :, :, 2:end, :];
trainset_pars = trainset_pars[:, 1:end-1, :];


trainset_init_distribution = reshape(trainset_init_distribution, H, W, C, (num_steps-1)*num_train);
trainset_target_distribution = reshape(trainset_target_distribution, H, W, C, (num_steps-1)*num_train);
trainset_pars_distribution = reshape(trainset_pars, num_pars, (num_steps-1)*num_train);


##### Hyperparameters #####

embedding_dims = 256;
batch_size = 8;
learning_rate = 5e-4;
weight_decay = 1e-8;
num_epochs = 200000;
num_samples = 9;

##### conditional SI model #####

velocity = AttnParsConvNextUNet(
    (H, W); 
    in_channels=C, 
    channels=[16, 32, 64, 128], 
    embedding_dims=embedding_dims, 
    block_depth=2,
    diffusion_multiplier=0.1f0,
    pars_dim=pars_dim,
    len_history=1,
    use_attention_in_layer=[false, false, true, true],
)

gamma = t -> 1f0 .- t;
dgamma_dt = t -> -ones(size(t));
diffusion_coefficient = t -> sqrt.((3f0 .- t) .* (1f0 .- t));

alpha = t -> 1f0 .- t; 
beta = t -> t.^2;
dalpha_dt = t -> -1f0;
dbeta_dt = t -> 2f0 .* t;
model = FollmerStochasticInterpolant(
    velocity, 
    interpolant=Interpolant(alpha, beta, dalpha_dt, dbeta_dt),
    gamma=Gamma(gamma, dgamma_dt),
    diffusion_coefficient=DiffusionCoefficient(diffusion_coefficient),
    diffusion_multiplier=0.1f0,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

model_save_dir = "trained_models/forecasting_model";

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Load checkpoint #####
continue_training = false;
continue_epoch = 225;
if continue_training
    ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/checkpoint_epoch_$continue_epoch.bson") .|> dev;
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
    num_test_paths=5,
    model_save_dir=model_save_dir,
    num_epochs=num_epochs,
    batch_size=batch_size,
    normalize_data=normalize_data,
    mask=mask,
    rng=rng,
    dev=dev
)


# test_init_condition = testset[:, :, :, 1:1, 1:1] |> dev;
# test_pars = testset_pars[:, 1:1, :] |> dev;
# num_test_steps = size(testset, 4) |> dev;
num_test_paths = 5 |> dev;


num_test_trajectories = size(testset)[end];
num_channels = size(testset, 3);
num_test_steps = size(testset, 4);

st_ = Lux.testmode(st);

if !isnothing(normalize_data)
    x_true = normalize_data.inverse_transform(testset)
else
    x_true = testset
end;

if !isnothing(mask)
    x_true = x_true .* mask
    num_non_obstacle_grid_points = sum(mask)
else
    num_non_obstacle_grid_points = size(x_true)[1] * size(x_true)[2]
end;

pathwise_MSE = []
mean_MSE = []
x = zeros(H, W, C, num_test_steps, num_test_paths) |> dev;
for i = 1:num_test_trajectories

    test_init_condition = testset[:, :, :, 1:1, i]
    test_pars = testset_pars[:, 1:1, i]

    x = compute_multiple_SDE_steps(
        init_condition=test_init_condition,
        parameters=test_pars,
        num_physical_steps=num_test_steps,
        num_generator_steps=40,
        num_paths=num_test_paths,
        model=model,
        ps=ps,
        st=st_,
        rng=rng,
        dev=dev,
        mask=mask,
    )


    if !isnothing(normalize_data)
        x = normalize_data.inverse_transform(x)
    end

    if !isnothing(mask)
        x = x .* mask
    end

    error_i = 0
    for j = 1:num_test_paths
        error_i += sum((x[:, :, :, :, j] - x_true[:, :, :, :, i]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
    end
    error_i = error_i / num_test_paths

    push!(pathwise_MSE, error_i)

    x_mean = mean(x, dims=5)[:, :, :, :, 1]
    x_std = std(x, dims=5)[:, :, :, :, 1]

    MSE = sum((x_mean - x_true[:, :, :, :, i]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
    push!(mean_MSE, MSE)
end;

println("Mean of pathwise MSE: ", mean(pathwise_MSE))
println("Std of pathwise MSE: ", std(pathwise_MSE))

println("Mean of mean MSE (SDE): ", mean(mean_MSE))
println("Std of mean MSE (SDE): ", std(mean_MSE))

x_mean = mean(x, dims=5)[:, :, :, :, 1];
x_std = std(x, dims=5)[:, :, :, :, 1];

x_true = x_true[:, :, :, :, num_test_trajectories];

save_path = @sprintf("output/tra_long_sde_SI_test.gif")

preds_to_save = (x_true[:, :, 4, :], x_mean[:, :, 4, :], Float16.(x_mean[:, :, 4, :]-x_true[:, :, 4, :]), Float16.(x_std[:, :, 4, :]), x[:, :, 4, :, 1], x[:, :, 4, :, 2], x[:, :, 4, :, 3], x[:, :, 4, :, 4]);
create_gif(preds_to_save, save_path, ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"])

CUDA.reclaim()
GC.gc()



# x = compute_multiple_ODE_steps(
#     init_condition=test_init_condition,
#     parameters=test_pars,
#     num_physical_steps=num_test_steps,
#     num_generator_steps=25,
#     model=model,
#     ps=ps,
#     st=st,
#     dev=dev,
#     mask=mask
# );

# num_channels = size(x, 3)

# if !isnothing(normalize_data)
#     x = normalize_data.inverse_transform(x)
#     testset = normalize_data.inverse_transform(testset)
# end

# if !isnothing(mask)
#     x = x .* mask
#     testset = testset .* mask

#     num_non_obstacle_grid_points = sum(mask)
# end

# MSE = sum((x[:, :, :, :, 1] - testset[:, :, :, :, 1]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
# println("MSE (ODE): ", MSE)

# # println("Time stepping error (ODE): ", mean(error))

# x = x[:, :, 4, :, 1]
# x_true = testset[:, :, 4, :, 1]


# save_path = @sprintf("output/ode_SI_%i.gif", 1)

# preds_to_save = (x_true, x, x-x_true)
# create_gif(preds_to_save, save_path, ["True", "Pred", "Error"])



# x = compute_multiple_SDE_steps(
#     init_condition=test_init_condition,
#     parameters=test_pars,
#     num_physical_steps=num_test_steps,
#     num_generator_steps=25,
#     num_paths=num_test_paths,
#     model=model,
#     ps=ps,
#     st=st,
#     rng=rng,
#     dev=dev,
#     mask=mask
# );


# num_channels = size(x, 3)
                
# if !isnothing(normalize_data)
#     x = normalize_data.inverse_transform(x)
#     testset = normalize_data.inverse_transform(testset)
# end

# if !isnothing(mask)
#     x = x .* mask
#     testset = testset .* mask

#     num_non_obstacle_grid_points = sum(mask)
# end

# MSE = 0.0f0
# for i = 1:num_test_paths
#     MSE += sum((x[:, :, :, :, i] - testset[:, :, :, :, 1]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
# end

# MSE /= num_test_paths
# println("MSE over paths (SDE): ", MSE)

# x_true = testset[:, :, :, :, 1]

# x_mean = mean(x, dims=5)[:, :, :, :, 1]
# x_std = std(x, dims=5)[:, :, :, :, 1]

# MSE = sum((x_mean - x_true).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
# println("MSE over mean (SDE): ", MSE)

# save_path = @sprintf("output/sde_SI_%i.gif", 1)

# preds_to_save = (x_true[:, :, 4, :], x_mean[:, :, 4, :], Float16.(x_mean[:, :, 4, :]-x_true[:, :, 4, :]), Float16.(x_std[:, :, 4, :]), x[:, :, 4, :, 1], x[:, :, 4, :, 2], x[:, :, 4, :, 3], x[:, :, 4, :, 4])
# create_gif(preds_to_save, save_path, ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"])



# x = trainset_init_distribution[:, :, :, 1:4] |> dev;
# x0 = x  |> dev;
# pars = randn(rng, (num_pars, 4)) |> dev;
# t = rand(rng, Float32, (1, 1, 1, 4)) |> dev;

# out, st = model.velocity((x, x0, pars, t), ps, st);



# if "inc" in l and "mixed" in l:
#     # ORDER (fields): velocity (x,y), --, pressure, ORDER (params): rey, --, --
#     self.normMean = np.array([0.444969, 0.000299, 0, 0.000586, 550.000000, 0, 0], dtype=np.float32)
#     self.normStd =  np.array([0.206128, 0.206128, 1, 0.003942, 262.678467, 1, 1], dtype=np.float32)

# if "tra" in l and "mixed" in l:
#     # ORDER (fields): velocity (x,y), density, pressure, ORDER (params): rey, mach, --
#     self.normMean = np.array([0.560642, -0.000129, 0.903352, 0.637941, 10000.000000, 0.700000, 0], dtype=np.float32)
#     self.normStd =  np.array([0.216987, 0.216987, 0.145391, 0.119944, 1, 0.118322, 1], dtype=np.float32)

# if "iso" in l and "single" in l:
#     # ORDER (fields): velocity (x,y,z), pressure, ORDER (params): --, --, --
#     self.normMean = np.array([-0.054618, -0.385225, -0.255757, 0.033446, 0, 0, 0], dtype=np.float32)
#     self.normStd =  np.array([0.539194, 0.710318, 0.510352, 0.258235, 1, 1, 1], dtype=np.float32)





###### load training set #####
# trainset = MNIST(:train)
# # trainset = CIFAR10(:train)
# trainset = trainset[1:num_train];
# trainset = trainset.features;
# trainset = imresize(trainset, (32, 32, num_train));
# trainset = reshape(trainset, 32, 32, 1, num_train);

# start_time = 100;
# num_steps = 150;
# num_train = 5;
# skip_steps = 2;
# data_train = load("data/data_train.jld2", "data_train");
# #data_train[1].data[1].u[200][2]
# H, W = size(data_train[1].data[1].u[1][1]).-2;
# C = 2;
# trainset = zeros(H, W, C, num_steps, num_train); # Height, Width, Channels, num_steps, num_train
# for i in 1:num_train
#     counter = 1;
#     for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
#         trainset[:, :, 1, counter, i] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
#         trainset[:, :, 2, counter, i] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]

#         trainset[:, :, :, counter, i] = trainset[:, :, :, counter, i] ./ norm(trainset[:, :, :, counter, i]);

#         counter += 1
#     end
# end

# trainset[:, :, 1, :, :] = (trainset[:, :, 1, :, :] .- mean(trainset[:, :, 1, :, :])) ./ std(trainset[:, :, 1, :, :]);
# trainset[:, :, 2, :, :] = (trainset[:, :, 2, :, :] .- mean(trainset[:, :, 2, :, :])) ./ std(trainset[:, :, 2, :, :]);

# # min max normalization
# # trainset = (trainset .- minimum(trainset)) ./ (maximum(trainset) - minimum(trainset));

# x = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2);

# create_gif((x, x), "HF.gif", ("lol", "lol"))










# velocity = ConditionalUNet(
#     image_size; 
#     in_channels=in_channels,
#     channels=channels, 
#     block_depth=block_depth,
#     min_freq=min_freq, 
#     max_freq=max_freq, 
#     embedding_dims=embedding_dims,
# )
# velocity =  ConvNextUNet(
#     image_size; 
#     in_channels=in_channels,
#     channels=channels, 
#     block_depth=block_depth,
#     min_freq=min_freq, 
#     max_freq=max_freq, 
#     embedding_dims=embedding_dims
# )
# velocity = DitParsConvNextUNet(
#     image_size; 
#     in_channels=in_channels,
#     channels=channels, 
#     block_depth=block_depth,
#     min_freq=min_freq, 
#     max_freq=max_freq, 
#     embedding_dims=embedding_dims,
#     pars_dim=1
# )
# velocity = ConditionalDiffusionTransformer(
#     image_size;
#     in_channels=in_channels, 
#     patch_size=(8, 8),
#     embed_dim=256, 
#     depth=4, 
#     number_heads=8,
#     mlp_ratio=4.0f0, 
#     dropout_rate=0.1f0, 
#     embedding_dropout_rate=0.1f0,
# )