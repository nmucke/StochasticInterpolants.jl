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

rng = Random.default_rng()
Random.seed!(rng, 0)


# load training set
trainset = MNIST(:train)

heatmap(trainset[89].features)


in_channels = 1
out_channels = 1
kernel_size = (3, 3)
embedding_dims = 4

H = 28
W = 28
image_size = (H, W)

t = randn(rng, Float32, 1, 1, 1, 1) .+ 10;
unet = UNet(image_size; in_channels=1, channels=[2, 4, 8], embedding_dims=embedding_dims)
ps, st = Lux.setup(rng, unet);

x = reshape(trainset[89].features, 28, 28, 1, 1)

lal, st = unet((x, t), ps, st);

opt = Optimisers.Descent(0.01f0)

function mse(model, ps, st, X)
    X_pred, st_new = model((X, t), ps, st)
    return sum(abs2, X_pred .- X), st_new
end

for i in 1:100
    # Compute the gradient using the pullback API to update the states
    (loss, st), pb_f = Zygote.pullback(mse, unet, ps, st, x)
    # We pass nothing as the seed for `st`, since we don't want to propagate any gradient
    # for st
    gs = pb_f((one(loss), nothing))[1]
    # Update model parameters
    # `Optimisers.update` can be used if mutation is not desired
    opt_state, ps = Optimisers.update!(opt_state, ps, gs)
    (i % 10 == 1 || i == 100) && println(lazy"Loss Value after $i iterations: $loss")
end