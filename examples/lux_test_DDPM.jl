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

rng = Random.default_rng()
Random.seed!(rng, 0)

# Get the device determined by Lux
dev = gpu_device()

##### Hyperparameters #####
num_train = 10000;
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
timesteps = 500;
noise_scheduling = get_noise_scheduling(
    timesteps,
    0.0001,
    0.02,
    "linear",
    dev
);

dev = gpu

in_channels = 1;
out_channels = 1;
kernel_size = (3, 3);
embedding_dims = 16;

H = 32;
W = 32;
image_size = (H, W);

unet = UNet(image_size; in_channels=1, channels=[16, 32, 64, 128], embedding_dims=embedding_dims);
ps, st = Lux.setup(rng, unet) .|> dev;


opt = Optimisers.Adam(1e-4, (0.9f0, 0.99f0), 1e-8);
opt_state = Optimisers.setup(opt, ps);

for epoch in 1:1000
    running_loss = 0.0
    for i in 1:batch_size:size(trainset)[end]  

        if i + batch_size - 1 > size(trainset)[end]
            break
        end

        x = trainset[:, :, :, i:i+batch_size-1] |> dev;
        t = rand(rng, 1:timesteps, (batch_size,)) |> dev;

        # loss, st = get_loss(x, t, noise_scheduling, unet, ps, st, rng, dev)

        (loss, st), pb_f = Zygote.pullback(
            p -> get_loss(x, t, noise_scheduling, unet, p, st, rng, dev), 
            ps
        );

        running_loss += loss

        gs = pb_f((one(loss), nothing))[1];
        
        opt_state, ps = Optimisers.update!(opt_state, ps, gs)
    end

    running_loss /= floor(Int, size(trainset)[end] / batch_size)
    
    (epoch % 5 == 1 || epoch == 100) && println(lazy"Loss Value after $epoch iterations: $running_loss")

    if epoch % 5 == 1 || epoch == 100

        x = randn(rng, Float32, 32, 32, 1, 1) |> dev
        for i in 1:timesteps
            t = [i] |> dev
            x, st = sample_timestep(x, t, unet, noise_scheduling, ps, st, rng, dev)
        end

        x = clamp!(x, 0, 1)
        
        x = Array(x)
        
        heatmap(x[:, :, 1, 1])
        #save(joinpath(output_dir, @sprintf("img_%.3d_epoch_%.4d.png", i, epoch)), img)

        save_dir = joinpath("output/train/images/", @sprintf("img_%i.png", epoch))
        savefig(save_dir)
    end

end

x = randn(rng, Float32, 32, 32, 1, 1) |> dev
for i in 1:timesteps
    t = [i] |> dev
    img, st = sample_timestep(
        x, 
        t,
        unet,
        noise_scheduling,
        ps,
        st,
        dev
    )
end

x = Array(x)
heatmap(x[:,:,1,1])






function heatgif(A)
    p = heatmap(A[:,:,1,1])
    anim = @animate for i=1:size(A,4)
        heatmap!(p[1], A[:,:,1,i])
    end
    return anim
end

anim = heatgif(x_noisy)
gif(anim, "anim.gif", fps = 15)
