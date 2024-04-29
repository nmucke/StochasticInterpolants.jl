using Lux
using Random
using Images
using Augmentor
using MLUtils
using Optimisers
using Statistics
using ProgressBars
using Zygote
using CUDA
using BSON
using Comonicon
using Printf
using Plots


"""
    save_as_png

Save a batch of images as PNG files.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function save_as_png(
    images::AbstractArray{T, 4}, 
    output_dir,
    epoch
) where {T <: AbstractFloat}

    #img = @view images[:, :, :, 1]
    #img = colorview(Gray, permutedims(img, (3, 1, 2)))

    heatmap(images[:, :, 1, 1])
    #save(joinpath(output_dir, @sprintf("img_%.3d_epoch_%.4d.png", i, epoch)), img)

    save_dir = joinpath(output_dir, @sprintf("img_%i.png", epoch))
    savefig(save_dir)


    # for i in axes(images, 4)
    #     img = @view images[:, :, :, i]
    #     img = colorview(Gray, permutedims(img, (3, 1, 2)))
    #     save(joinpath(output_dir, @sprintf("img_%.3d_epoch_%.4d.png", i, epoch)), img)
    # end

end


"""
    compute_loss

Compute the loss of the DenoisingDiffusionImplicitModel.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function compute_loss(
    ddim::DenoisingDiffusionImplicitModel{T}, 
    images::AbstractArray{T, 4},
    rng::AbstractRNG, 
    ps, 
    st::NamedTuple
) where {T <: AbstractFloat}

    (noises, images, pred_noises, pred_images), st = ddim((images, rng), ps, st)
    noise_loss = mean(abs.(pred_noises - noises))
    image_loss = mean(abs.(pred_images - images))
    loss = noise_loss + image_loss

    return loss, st
end

"""
    train_step

Perform a training step for the DenoisingDiffusionImplicitModel.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function train_step(
    ddim::DenoisingDiffusionImplicitModel{T}, 
    images::AbstractArray{T, 4},
    rng::AbstractRNG, 
    ps, 
    st::NamedTuple,
    opt_st::NamedTuple
) where {T <: AbstractFloat}

    (loss, st), back = Zygote.pullback(
        p -> compute_loss(ddim, images, rng, p, st), 
        ps
    )
    gs = back((one(loss), nothing))[1]

    opt_st, ps = Optimisers.update(opt_st, ps, gs)

    return loss, ps, st, opt_st
end


function train_DDIM(; 
    dataset,#::String, 
    ddim::DenoisingDiffusionImplicitModel,
    epochs::Int=1, 
    image_size::Int=64,
    rng::AbstractRNG=Random.MersenneTwister(1234),
    batch_size::Int=64, 
    # learning_rate::Float64=1e-3,
    # weight_decay::Float64=1e-4, 
    val_diffusion_steps::Int=3,
    checkpoint_interval::Int=5, 
    output_dir::String="output/train",
    dev::Function=gpu,
    ps, st, opt_st
    # model hyper params
    # channels::Vector{Int}=[32, 64, 96, 128], 
    # block_depth::Int=2,
    # min_freq::Float32=1.0f0, 
    # max_freq::Float32=1000.0f0,
    # embedding_dims::Int=32, 
    # min_signal_rate::Float32=0.02f0,
    # max_signal_rate::Float32=0.95f0
)
    # rng = Random.MersenneTwister()
    # Random.seed!(rng, 1234)

    image_dir = joinpath(output_dir, "images")
    ckpt_dir = joinpath(output_dir, "ckpt")
    mkpath(image_dir)
    mkpath(ckpt_dir)

    # println("Preparing dataset.")
    # ds = ImageDataset(dataset_dir, x -> preprocess_image(x, image_size), true)
    # data_loader = DataLoader(ds; batchsize=batchsize, partial=false, collate=true,
    #             parallel=true, rng=rng, shuffle=true)

    # println("Preparing DDIM.")
    # ddim = DenoisingDiffusionImplicitModel((image_size, image_size); channels=channels,
    #                         block_depth=block_depth, min_freq=min_freq,
    #                         max_freq=max_freq, embedding_dims=embedding_dims,
    #                         min_signal_rate=min_signal_rate,
    #                         max_signal_rate=max_signal_rate)
    # ps, st = Lux.setup(rng, ddim) .|> gpu

    # println("Set optimizer.")
    # opt = AdamW(learning_rate, (9.0f-1, 9.99f-1), weight_decay)
    # opt_st = Optimisers.setup(opt, ps) |> gpu

    rng_gen = Random.MersenneTwister()
    Random.seed!(rng_gen, 0)

    println("Training.")
    iter = ProgressBar(1:epochs)
    for epoch in iter
        losses = []
        #iter = ProgressBar(data_loader)

        # Shuffle the dataset
        #dataset = shuffle(rng_gen, dataset, dims=4)

        st = Lux.trainmode(st)
        for i in 1:batch_size:size(dataset)[end]

            if i + batch_size - 1 > size(dataset)[end]
                continue
            end

            x = dataset[:, :, :, i:i+batch_size-1] |> dev;

            loss, ps, st, opt_st = train_step(ddim, x, rng, ps, st, opt_st)
            push!(losses, loss)
            set_description(iter, "Epoch: $(epoch) Loss: $(mean(losses))")
        end

        st = Lux.testmode(st)
        generated_images, _ = generate(
            ddim, 
            rng_gen, #Lux.replicate(rng_gen), # to get inference on the same noises
            (image_size, image_size, size(dataset)[3], 10), 
            val_diffusion_steps,
            ps, 
            st; 
            save_each_step=false,
            dev=dev
        )
    
        generated_images = generated_images |> cpu
        save_as_png(generated_images, image_dir, epoch)

        # if epoch % checkpoint_interval == 0
        #     save_checkpoint(ps, st, opt_st, ckpt_dir, epoch)
        # end
    end

    return ps, st

end