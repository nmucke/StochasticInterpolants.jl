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


# data = npzread("data/128_inc/128_inc/sim_000000/pressure_000000.npz")

H = 64;
W = 128;
C = 4;
num_pars = 1;
num_train = 35;

start_time = 1;
num_steps = 50;
skip_steps = 4;

# data_folder = "data/128_inc/128_inc";
data_folder = "data/128_tra";

trainset = zeros(H, W, C, num_steps, num_train);
trainset_pars = zeros(num_pars, num_steps, num_train);
for i in 0:num_train-1
    # for j in 0:num_steps-1
    counter = 1;
    for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
        trajectory = lpad(string(i), 6, '0')
        timestep = lpad(string(j), 6, '0')
        pressure = npzread("$(data_folder)/sim_$(trajectory)/pressure_$(timestep).npz")
        pressure = pressure["arr_0"]
        pressure = permutedims(pressure, (3, 2, 1))

        velocity = npzread("$(data_folder)/sim_$(trajectory)/velocity_$(timestep).npz")
        velocity = velocity["arr_0"]
        velocity = permutedims(velocity, (3, 2, 1))

        density = npzread("$(data_folder)/sim_$(trajectory)/density_$(timestep).npz")
        density = density["arr_0"]
        density = permutedims(density, (3, 2, 1))

        data = cat(pressure, velocity, density, dims=3)

        trainset[:, :, :, counter, i+1] = data[:, :, :]
        
        pars = JSON.parsefile("$(data_folder)/sim_$(trajectory)/src/description.json")
        trainset_pars[1, counter, i+1] = pars["Mach Number"]

        counter += 1
    end
end

trainset[:, :, 1, :, :] = (trainset[:, :, 1, :, :] .- mean(trainset[:, :, 1, :, :])) ./ std(trainset[:, :, 1, :, :]);
trainset[:, :, 2, :, :] = (trainset[:, :, 2, :, :] .- mean(trainset[:, :, 2, :, :])) ./ std(trainset[:, :, 2, :, :]);
trainset[:, :, 3, :, :] = (trainset[:, :, 3, :, :] .- mean(trainset[:, :, 3, :, :])) ./ std(trainset[:, :, 3, :, :]);
trainset[:, :, 4, :, :] = (trainset[:, :, 4, :, :] .- mean(trainset[:, :, 4, :, :])) ./ std(trainset[:, :, 4, :, :]);

x1 = sqrt.(trainset[:, :, 2, :, 1].^2 + trainset[:, :, 3, :, 1].^2); 
x2 = sqrt.(trainset[:, :, 2, :, 30].^2 + trainset[:, :, 3, :, 30].^2);


A = (x1, x2);
plot_titles = ["1", "2"];
create_gif(A, "HF.gif", plot_titles)

# Set the seed

rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();

##### Hyperparameters #####
kernel_size = (5, 5);
embedding_dims = 128;
batch_size = 8;
learning_rate = 5e-4;
weight_decay = 1e-6;
num_epochs = 200000;

num_samples = 9;

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

# Divide the training set into initial and target distributions
trainset_init_distribution = trainset[:, :, :, 1:end-1, :];
trainset_target_distribution = trainset[:, :, :, 2:end, :];
trainset_pars = trainset_pars[:, 1:end-1, :];

trainset_init_distribution = reshape(trainset_init_distribution, H, W, C, (num_steps-1)*num_train);
trainset_target_distribution = reshape(trainset_target_distribution, H, W, C, (num_steps-1)*num_train);
trainset_pars_distribution = reshape(trainset_pars, num_pars, (num_steps-1)*num_train);

H, W, C = size(trainset_init_distribution)[1:3];

image_size = (H, W);
in_channels = C;
out_channels = C;


##### conditional SI model #####
model = ForecastingStochasticInterpolant(
    image_size; 
    in_channels=in_channels, 
    channels=[16, 32, 64, 128], 
    embedding_dims=embedding_dims, 
    block_depth=2,
    num_steps=num_steps,
    dev=dev
)
ps, st = Lux.setup(rng, model) .|> dev;

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Train stochastic interpolant #####
ps, st = train_stochastic_interpolant(
    model,
    ps,
    st,
    opt_state,
    trainset_target_distribution,
    num_epochs,
    batch_size, 
    num_steps,
    rng,
    trainset_init_distribution,
    trainset_pars_distribution,
    true,
    dev
)




# x = trainset_init_distribution[:, :, :, 1:4] |> dev;
# x0 = x  |> dev;
# pars = randn(rng, (num_pars, 4)) |> dev;
# t = rand(rng, Float32, (1, 1, 1, 4)) |> dev;

# out, st = model.velocity((x, x0, pars, t), ps, st);