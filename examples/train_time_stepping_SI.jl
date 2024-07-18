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
# options are pars_extrapolation, pars_interpolation, long_rollouts
test_args = "pars_interpolation";

# Load the test case configuration
test_case_config = YAML.load_file("configs/test_cases/$test_case.yml");

# Get data path
data_folder = test_case_config["data_folder"];

# Get dimensions of state and parameter spaces
H = test_case_config["state_dimensions"]["height"];
W = test_case_config["state_dimensions"]["width"];
C = test_case_config["state_dimensions"]["channels"];
num_pars = length(test_case_config["parameter_dimensions"]);

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
normalize_data = NormalizeData(
    test_case_config["norm_mean"], 
    test_case_config["norm_std"]
);
trainset = normalize_data.transform(trainset);
testset = normalize_data.transform(testset);

x1, p1 = trainset[:, :, 4, :, 1], trainset_pars[1, 1, 1];
x2, p2 = trainset[:, :, 4, :, 2], trainset_pars[1, 1, 2];
x3, p3 = trainset[:, :, 4, :, 3], trainset_pars[1, 1, 3];
# x4, p4 = trainset[:, :, 4, :, 32], trainset_pars[1, 1, 32];
create_gif((x1, x2, x3), "HF.gif", ("Ma = $p1", "Ma = $p2", "Ma = $p3"))
# create_gif((x1, x2, x3, x4), "HF.gif", ("Ma = $p1", "Ma = $p2", "Ma = $p3", "Ma = $p4"))

# Divide the training set into initial and target distributions
trainset_init_distribution = trainset[:, :, :, 1:end-1, :];
trainset_target_distribution = trainset[:, :, :, 2:end, :];
trainset_pars = trainset_pars[:, 1:end-1, :];

trainset_init_distribution = reshape(trainset_init_distribution, H, W, C, (num_steps-1)*num_train);
trainset_target_distribution = reshape(trainset_target_distribution, H, W, C, (num_steps-1)*num_train);
trainset_pars_distribution = reshape(trainset_pars, num_pars, (num_steps-1)*num_train);


##### Hyperparameters #####
embedding_dims = 128;
batch_size = 8;
learning_rate = 5e-4;
weight_decay = 1e-6;
num_epochs = 200000;
num_samples = 9;

##### conditional SI model #####
model = ForecastingStochasticInterpolant(
    (H, W); 
    in_channels=C, 
    channels=[16, 32, 64, 128], 
    embedding_dims=embedding_dims, 
    block_depth=2,
    num_steps=num_steps,
    diffusion_multiplier=1.0f0,
    dev=dev
)
ps, st = Lux.setup(rng, model) .|> dev;

model_save_dir = "trained_models/forecasting_model";

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

continue_training = false;
continue_epoch = 0;
##### Load checkpoint #####
if continue_training
    ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/checkpoint_epoch_$continue_epoch.bson");
end
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
    rng=rng,
    dev=dev
)




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