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

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();

##### Hyperparameters #####
kernel_size = (5, 5);
embedding_dims = 4;
batch_size = 8;
learning_rate = 1e-3;
weight_decay = 1e-10;
num_epochs = 1000;
num_samples = 9;

###### load training set #####
# trainset = MNIST(:train)
# # trainset = CIFAR10(:train)
# trainset = trainset[1:num_train];
# trainset = trainset.features;
# trainset = imresize(trainset, (32, 32, num_train));
# trainset = reshape(trainset, 32, 32, 1, num_train);

start_time = 200;
num_steps = 50;
num_train = 5;
skip_steps = 4;
data_train = load("data/data_train.jld2", "data_train");
#data_train[1].data[1].u[200][2]
H, W = size(data_train[1].data[1].u[1][1]).-2;
trainset = zeros(H, W, 2, num_steps, num_train); # Height, Width, Channels, num_steps, num_train
for i in 1:num_train
    counter = 1;
    for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
        trainset[:, :, 1, counter, i] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
        trainset[:, :, 2, counter, i] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]

        trainset[:, :, :, counter, i] = trainset[:, :, :, counter, i] ./ norm(trainset[:, :, :, counter, i]);

        counter += 1
    end
end

# trainset[:, :, 1, :, :] = (trainset[:, :, 1, :, :] .- mean(trainset[:, :, 1, :, :])) ./ std(trainset[:, :, 1, :, :]);
# trainset[:, :, 2, :, :] = (trainset[:, :, 2, :, :] .- mean(trainset[:, :, 2, :, :])) ./ std(trainset[:, :, 2, :, :]);

# min max normalization
# trainset = (trainset .- minimum(trainset)) ./ (maximum(trainset) - minimum(trainset));

x = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2);

# create_gif((x), "HF.gif", plot_titles=("lol"))

# Divide the training set into initial and target distributions
trainset_init_distribution = trainset[:, :, :, 1:end-1, :];
trainset_target_distribution = trainset[:, :, :, 2:end, :];

trainset_init_distribution = reshape(trainset_init_distribution, H, W, 2, (num_steps-1)*num_train);
trainset_target_distribution = reshape(trainset_target_distribution, H, W, 2, (num_steps-1)*num_train);

H, W, C = size(trainset_init_distribution)[1:3];
image_size = (H, W);
in_channels = C;
out_channels = C;



#trainset = reshape(trainset, H, W, C, num_train);

##### conditional SI model #####
model = ForecastingStochasticInterpolant(
    image_size; 
    in_channels=C, 
    channels=[8, 16, 32], 
    embedding_dims=embedding_dims, 
    block_depth=1,
    num_steps=num_steps,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Train stochastic interpolant #####
num_samples = num_steps;
ps, st = train_stochastic_interpolant(
    model,
    ps,
    st,
    opt_state,
    trainset_target_distribution,
    num_epochs,
    batch_size, 
    num_samples,
    rng,
    trainset_init_distribution,
    true,
    dev
)

