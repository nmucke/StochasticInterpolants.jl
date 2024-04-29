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

##### Model #####
ddim = DenoisingDiffusionImplicitModel(
    image_size; 
    in_channels=1, 
    channels=[4, 8, 16, 32], 
    embedding_dims=embedding_dims,
    block_depth=2
);
ps, st = Lux.setup(rng, ddim) .|> dev;

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (9.0f-1, 9.99f-1), weight_decay);
opt_state = Optimisers.setup(opt, ps) |> dev;

##### Training #####
ps, st = train_DDIM(
    dataset=trainset,
    ddim=ddim,
    epochs=1000, 
    image_size=H,
    rng=Random.MersenneTwister(1234),
    batch_size=batch_size, 
    val_diffusion_steps=100,
    checkpoint_interval=5, 
    output_dir="output/train",
    dev=dev,
    ps=ps, 
    st=st, 
    opt_st=opt_state
);

generated_images, images_each_step = generate(
    ddim, 
    rng, 
    (H, W, 1, 1), 
    100, 
    ps, 
    st; 
    save_each_step=true,
    dev=dev
);

generated_images = Array(generated_images);
heatmap(generated_images[:, :, 1, 1])



