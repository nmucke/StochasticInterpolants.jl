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

rng = Random.default_rng()
Random.seed!(rng, 0)

# Get the device determined by Lux
dev = gpu_device()


# load training set
trainset = MNIST(:train)
trainset = trainset[1:1000];

x = reshape(trainset.features[:, :, 32], 28, 28);

heatmap(x)


in_channels = 1;
out_channels = 1;
kernel_size = (3, 3);
embedding_dims = 4;

H = 28;
W = 28;
image_size = (H, W);

batch_size = 32;

unet = UNet(image_size; in_channels=1, channels=[8, 16, 32], embedding_dims=embedding_dims);
ps, st = Lux.setup(rng, unet) .|> dev;


opt = Optimisers.Adam(0.001f0);
opt_state = Optimisers.setup(opt, ps);


function mse(model, ps, st, X_noise, X)
    t = ones(Float32, 1, 1, 1, size(X)[end])  |> dev;
    X_pred, st_new = model((X_noise, t), ps, st)
    return sum(abs, X_pred .- X), st_new
end

trainset = trainset |> dev;

for epoch in 1:1000

    running_loss = 0.0

    for i in 1:batch_size:size(trainset.features)[end]

        if i + batch_size - 1 > size(trainset.features)[end]
            break
        end

        x = reshape(trainset.features[:, :, i:i+batch_size-1], 28, 28, 1, batch_size);

        noise = 0.1f0 * randn(rng, Float32, size(x)) |> dev;

        # Add noise to the input
        x_noise = x .+ noise

        (loss, st), pb_f = Zygote.pullback(p -> mse(unet, p, st, x_noise, x), ps);

        running_loss += loss

        gs = pb_f((one(loss), nothing))[1];
        
        opt_state, ps = Optimisers.update!(opt_state, ps, gs)

    end

    running_loss /= floor(Int, size(trainset.features)[end] / batch_size)

    #loss = mse(unet, ps, st, x)[1]
    
    (epoch % 5 == 1 || epoch == 100) && println(lazy"Loss Value after $epoch iterations: $running_loss")

end

x = reshape(trainset.features[:, :, 32], 28, 28, 1, 1);
x = Array(x);

heatmap(x[:, :, 1, 1])
x_noise = x .+ 0.1f0 * randn(rng, Float32, size(x));
heatmap(x_noise[:, :, 1, 1])

x_noise = x_noise |> dev;
t = ones(Float32, 1, 1, 1, size(x)[end]) |> dev;
x_pred, st = unet((x_noise, t), ps, st);

# move the output to the CPU
x_pred = Array(x_pred);
x_noise = Array(x_noise);

heatmap(x_pred[:, :, 1, 1])


heatmap(x_pred[:, :, 1, 1] .- x_noise[:, :, 1, 1])