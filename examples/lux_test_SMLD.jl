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
learning_rate = 1e-4;
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
    in_channels=C, 
    channels=[4, 8, 16, 32, 64], 
    embedding_dims=embedding_dims, 
    block_depth=2,
);
ps, st = Lux.setup(rng, model) .|> dev;


##### Optimizer #####
opt = Optimisers.Adam(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Train SMLD #####
num_samples = 9;
ps,st = train_diffusion_model(
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


# using DifferentialEquations
# using StochasticDiffEq
# using DiffEqFlux
# using StochasticDiffEq, DiffEqBase.EnsembleAnalysis, Random


# t = ones((1, 1, 1, num_samples)) |> dev
# x = randn(rng, Float32, model.unet.upsample.size..., model.unet.conv_in.in_chs, num_samples) |> dev
# x = x .* Float32.(model.marginal_probability_std(t))

# t_span = (0.0, 1.0)

# num_steps = 100
# timesteps = LinRange(1, model.eps, num_steps)
# step_size = Float32.(timesteps[1] - timesteps[2]) |> dev


# backward_drift_term(x, ps, t, st) = -model.diffusion_coefficient(t).^2 .* unet((x, t), ps, st)
# backward_diffusion_term(x, ps, t, st) = model.diffusion_coefficient(t)[1, 1, 1, :]


# backward_drift_term(x, p, t) = x#model.unet((x, reshape(t, 1, 1, 1, length(t))), ps, st)
# backward_diffusion_term(x, p, t) = t#model.diffusion_coefficient(t)[1, 1, 1, 1]


# drift = Chain(
#     model.unet,
#     x -> -model.diffusion_coefficient(t).^2 .* x
# )
# drift = Lux.Experimental.StatefulLuxLayer(model.unet, nothing, st)
# diffusion = Lux.Experimental.StatefulLuxLayer(model.unet, nothing, st)



# dudt(u, p, t) = -model.diffusion_coefficient(t).^2 .* drift((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)
# g(u, p, t) = model.diffusion_coefficient(t) .* ones(size(u)) |> dev   #diffusion((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

# ff = SDEFunction{false}(dudt, g)
# prob = SDEProblem{false}(ff, g, x, reverse(t_span), ps)

# sol = solve(
#     prob, 
#     save_everystep = false,
# )


# CUDA.reclaim()



# sol, st = (
#     solve(
#         prob; 
#         u0 = x, 
#         ps, 
#         sensealg = TrackerAdjoint(),     
#     ),
#     (; drift = drift.st, diffusion = diffusion.st)
# )




# x = x |> dev
# t = ones((1, 1, 1, num_samples)) |> dev
# drift((x, t), ps)














# function lorenz(du, u, p, t)
#     du = u
# end

# function σ_lorenz(du, u, p, t)
#     du = t
# end

# neuralsde = NeuralDSDE(
#     model.unet, 
#     model.unet, 
#     t_span, 
#     SOSRI(),
#     saveat = timesteps, 
#     reltol = 1e-1, 
#     abstol = 1e-1
# )

# prediction0, st1, st2 = neuralsde(x,ps,st,st)


























# prob_sde_lorenz = NeuralDSDE(lorenz, σ_lorenz, x, (0.0, 10.0))
# sol = solve(prob_sde_lorenz)
# plot(sol, idxs = (1, 2, 3))


# sde_problem = SDEProblem(
#     backward_drift_term,
#     backward_diffusion_term, 
#     x,
#     reverse(t_span), 
#     ps,
#     save_everystep = false,
# )
# sol = solve(sde_problem, dt=step_size)



# sde_problem = NeuralDSDE(
#     backward_drift_term,
#     backward_diffusion_term, 
#     reverse(t_span), 
#     SOSRI(); 
#     save_everystep = false,
#     reltol = 1e-3, 
#     abstol = 1e-3, 
#     save_start = false
# )

# p1 = Lux.ComponentArray(p)
# p2 = Lux.ComponentArray(p)

# prediction0, st1, st2 = sde_problem(u0,p,st1,st2)








# function sde_sampling(
#     model::ScoreMatchingLangevinDynamics,
#     ps::NamedTuple,
#     st::NamedTuple,
#     rng::AbstractRNG,
#     num_samples::Int,
#     num_steps::Int,
#     dev=gpu
# )

#     t = ones((1, 1, 1, num_samples)) |> dev
#     x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples) |> dev
#     x = x .* Float32.(marginal_probability_std(t))

#     t_span = (0.0, 1.0)

#     timesteps = LinRange(1, eps, num_steps)
#     step_size = Float32.(timesteps[1] - timesteps[2]) |> dev

#     sde_problem = NeuralDSDE(
#         model.backward_drift_term,
#         model.backward_diffusion_term, 
#         reverse(t_span), 
#         SOSRI(); 
#         save_everystep = false,
#         reltol = 1e-3, 
#         abstol = 1e-3, 
#         save_start = false
#     )



#     return x
# end



# t = ones((1, 1, 1, num_samples)) |> dev
# x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples) |> dev
# x = x .* Float32.(marginal_probability_std(t))
# timesteps = LinRange(1, eps, num_steps)# |> dev
# step_size = Float32.(timesteps[1] - timesteps[2]) |> dev

# problem = SDEProblem(
#     backward_drift_term, 
#     backward_diffusion_term, 
#     randn(rng, Float32, size(unet.upsample.size..., unet.conv_in.in_chs, num_samples)) .* marginal_probability_std(0.0), 
#     (1.0, 0.0)
# )

# dudt(u, ps, t, st) = backward_drift_term((u, 1-t), ps, st)
# g(u, ps, t) = backward_diffusion_term(u, ps, 1-t, st)
# prob = SDEProblem(dudt, g, x, tspan, nothing)
