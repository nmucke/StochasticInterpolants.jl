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
using Printf
using Statistics

Plots.default(linewidth=1, label=nothing, grid=false, tickfontsize=4, size = (1000, 700));

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();

##### Hyperparameters #####
num_train = 1000;
kernel_size = (5, 5);
embedding_dims = 4;
batch_size = 32;
learning_rate = 1e-3;
weight_decay = 1e-8;
timesteps = 100;
num_epochs = 10000;
num_samples = 9;

###### load training set #####
#trainset = MNIST(:train)
# trainset = CIFAR10(:train)
# trainset = trainset[1:num_train];
# trainset = trainset.features;


num_train = 200*5;
data_train = load("data/turbulence.jld2", "data_train");
#data_train[1].data[1].u[200][2]
H, W = size(data_train[1].data[1].u[1][1]).-2;
trainset = zeros(H, W, 2, 200*5);
for i in 1:5
    for j in 1:200
        trainset[:, :, 1, (i-1)*200 + j] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
        trainset[:, :, 2, (i-1)*200 + j] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]
    end
end

trainset[:, :, 1, :] = (trainset[:, :, 1, :] .- mean(trainset[:, :, 1, :])) ./ std(trainset[:, :, 1, :]);
trainset[:, :, 2, :] = (trainset[:, :, 2, :] .- mean(trainset[:, :, 2, :])) ./ std(trainset[:, :, 2, :]);

random_indices = rand(rng, 1:1000, 9);

x = trainset[:, :, :, random_indices];
x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2);
Plots.heatmap(x[:, :, 5])
plot_list = [];
for i in 1:9
    push!(plot_list, heatmap(x[:, :, i])); 
end

savefig(plot(plot_list..., layout=(3,3)), "lol.pdf");


H, W, C = size(trainset)[1:3];
image_size = (H, W);
in_channels = C;
out_channels = C;

trainset = reshape(trainset, H, W, C, num_train);

##### DDPM model #####
model = ScoreMatchingLangevinDynamics(
    image_size; 
    in_channels=C, channels=[4, 8, 16, 32, 64], embedding_dims=embedding_dims, block_depth=2,
);
ps, st = Lux.setup(rng, model) .|> dev;

##### Optimizer #####
opt = Optimisers.Adam(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Train SMLD #####
num_samples = 9;
train_diffusion_model(
    model,
    ps,
    st,
    opt_state,
    trainset,
    num_epochs,
    batch_size,
    num_samples,
    rng,
)


