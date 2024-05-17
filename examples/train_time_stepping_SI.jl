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
embedding_dims = 8;
batch_size = 32;
learning_rate = 1e-4;
weight_decay = 1e-6;
timesteps = 100;
num_epochs = 20000;
num_samples = 9;

###### load training set #####
# trainset = MNIST(:train)
# # trainset = CIFAR10(:train)
# trainset = trainset[1:num_train];
# trainset = trainset.features;
# trainset = imresize(trainset, (32, 32, num_train));
# trainset = reshape(trainset, 32, 32, 1, num_train);


num_steps = 200;
num_train = 5;
data_train = load("data/data_train.jld2", "data_train");
#data_train[1].data[1].u[200][2]
H, W = size(data_train[1].data[1].u[1][1]).-2;
trainset = zeros(H, W, 2, num_steps, num_train); # Height, Width, Channels, num_steps, num_train
for i in 1:num_train
    for j in 1:num_steps
        trainset[:, :, 1, j, i] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
        trainset[:, :, 2, j, i] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]
    end
end

trainset[:, :, 1, :, :] = (trainset[:, :, 1, :, :] .- mean(trainset[:, :, 1, :, :])) ./ std(trainset[:, :, 1, :, :]);
trainset[:, :, 2, :, :] = (trainset[:, :, 2, :, :] .- mean(trainset[:, :, 2, :, :])) ./ std(trainset[:, :, 2, :, :]);

x = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2);

# create_gif(x, "HF.gif")


# random_indices = rand(rng, 1:1000, 9);

# x = trainset[:, :, :, random_indices];
# x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2);
# Plots.heatmap(x[:, :, 5])
# plot_list = [];
# for i in 1:9
#     push!(plot_list, heatmap(x[:, :, i])); 
# end

# savefig(plot(plot_list..., layout=(3,3)), "hf_samples.pdf");

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

##### DDPM model #####
model = StochasticInterpolantModel(
    image_size; 
    in_channels=C, 
    channels=[8, 16, 32, 64, 128], 
    embedding_dims=embedding_dims, 
    block_depth=2,
    num_steps=400,
);
ps, st = Lux.setup(rng, model) .|> dev;

##### Optimizer #####
opt = Optimisers.Adam(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Train stochastic interpolant #####
num_samples = 9;
ps,st = train_stochastic_interpolant(
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
    true,
    gpu
)



# using DifferentialEquations
    
# #x = model.sample(num_samples, ps, st_, rng, dev)
# x = randn(rng, Float32, model.unet.upsample.size..., model.unet.conv_in.in_chs, num_samples) |> dev
# t_span = (0.0, 1.0)
# # t_span = Float32.(t_span)

# timesteps = LinRange(0, 1, 100)# |> dev
# dt = Float32.(timesteps[2] - timesteps[1]) |> dev


# stateful_drift_NN = Lux.Experimental.StatefulLuxLayer(model.unet, nothing, st);

# dudt(u, p, t) = stateful_drift_NN((u, t .* ones(Float32, 1, 1, 1, size(u)[end])) |> dev, p);


# #ff = ODEFunction{false}(dudt)
# prob = ODEProblem(dudt, x, t_span, ps)

# x = solve(
#     prob,
#     RK4(),
#     dt = dt,
#     save_everystep = false, 
#     adaptive = false
# )
# x = x[:, :, :, :, end]
