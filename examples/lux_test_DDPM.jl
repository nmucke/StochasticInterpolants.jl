using Lux
using StochasticInterpolants
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

rng = Random.default_rng()
Random.seed!(rng, 0)

# Get the device determined by Lux
dev = gpu_device()

##### Hyperparameters #####
num_train = 5000;
in_channels = 1;
out_channels = 1;
kernel_size = (3, 3);
embedding_dims = 4;
batch_size = 256;
learning_rate = 1e-3;
weight_decay = 1e-10;

######load training set #####
trainset = MNIST(:train)
trainset = trainset[1:num_train];
trainset = trainset.features;

# Reshape and move to device
trainset = reshape(trainset, 28, 28, 1, num_train);

# Interpolate onto 32 x 32
_trainset = zeros(Float32, 32, 32, 1, num_train);

for i in 1:num_train
    _trainset[:, :, 1, i] = imresize(trainset[:, :, 1, i], (32, 32));
end

trainset = _trainset;

H, W = size(trainset)[1:2];
image_size = (H, W);

##### Noise Scheduler #####
timesteps = 170;
noise_scheduling = get_noise_scheduling(
    timesteps,
    0.0001,
    0.02,
    "linear"
);

x0 = trainset[:, :, :, 1:5];
for i in 1:5
    x0[:, :, :, i] = x0[:, :, :, 1] 
end

t = [10, 40, 80, 120, 160];

x_noised, _ = forward_diffusion_sample(
    x0, 
    t,
    rng,
    noise_scheduling,
    cpu_device(),
);
