using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA

"""
    DenoisingDiffusionImplicitModel

A model that uses a UNet to denoise images using the diffusion model.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
struct DenoisingDiffusionImplicitModel{T <: AbstractFloat} <: Lux.AbstractExplicitContainerLayer{
    (:unet, :batchnorm)
}
    unet::UNet
    batchnorm::BatchNorm
    min_signal_rate::T
    max_signal_rate::T
end

function DenoisingDiffusionImplicitModel(
    image_size::Tuple{Int, Int};
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0,
    embedding_dims=32, 
    min_signal_rate=0.02f0,
    max_signal_rate=0.95f0
)
    unet = UNet(
        image_size; 
        in_channels=in_channels,
        channels=channels, 
        block_depth=block_depth, 
        min_freq=min_freq,
        max_freq=max_freq, 
        embedding_dims=embedding_dims
    )
    batchnorm = BatchNorm(in_channels)

    return DenoisingDiffusionImplicitModel(
        unet, batchnorm, min_signal_rate, max_signal_rate
    )
end

function (ddim::DenoisingDiffusionImplicitModel{T})(
    x::Tuple{AbstractArray{T, 4}, AbstractRNG}, 
    ps,
    st::NamedTuple,
    dev = gpu
) where {T <: AbstractFloat}

    images, rng = x
    images, new_st = ddim.batchnorm(images, ps.batchnorm, st.batchnorm)
    @set! st.batchnorm = new_st

    noises = randn(rng, eltype(images), size(images)...) |> dev

    diffusion_times = rand(rng, eltype(images), 1, 1, 1, size(images, 4)) |> dev
    noise_rates, signal_rates = diffusion_schedules(
        diffusion_times, 
        ddim.min_signal_rate,
        ddim.max_signal_rate
    )

    noisy_images = signal_rates .* images + noise_rates .* noises

    (pred_noises, pred_images), st = denoise(
        ddim, noisy_images, noise_rates, signal_rates, ps, st
    )

    return (noises, images, pred_noises, pred_images), st
end

"""
    diffusion_schedules

Generates noise with variable magnitude depending on time.
Noise is at minimum at t=0, and maximum at t=1.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function diffusion_schedules(
    diffusion_times::AbstractArray{T, 4}, 
    min_signal_rate::T,
    max_signal_rate::T
) where {T <: AbstractFloat}

    start_angle = acos(max_signal_rate)
    end_angle = acos(min_signal_rate)

    diffusion_angles = start_angle .+ (end_angle - start_angle) * diffusion_times

    # see Eq. (12) in 2010.02502 with sigma=0
    signal_rates = cos.(diffusion_angles) # sqrt{alpha_t}
    noise_rates = sin.(diffusion_angles) # sqrt{1-alpha_t}

    return noise_rates, signal_rates
end

"""
    denoise

Denoise images using the UNet.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function denoise(
    ddim::DenoisingDiffusionImplicitModel{T},
    noisy_images::AbstractArray{T, 4}, 
    noise_rates::AbstractArray{T, 4},
    signal_rates::AbstractArray{T, 4}, 
    ps,
    st::NamedTuple
) where {T <: AbstractFloat}

    pred_noises, new_st = ddim.unet(
        (noisy_images, noise_rates.^2), 
        ps.unet, 
        st.unet
    )
    @set! st.unet = new_st

    pred_images = (noisy_images - pred_noises .* noise_rates) ./ signal_rates

    return (pred_noises, pred_images), st
end

"""
    reverse_diffusion

Reverse the diffusion process to generate images.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function reverse_diffusion(
    ddim::DenoisingDiffusionImplicitModel{T},
    initial_noise::AbstractArray{T, 4}, 
    diffusion_steps::Int, 
    ps,
    st::NamedTuple; 
    save_each_step=false,
    dev = gpu
) where {T <: AbstractFloat}

    num_images = size(initial_noise, 4)
    step_size = convert(T, 1.0) / diffusion_steps

    next_noisy_images = initial_noise
    pred_images = nothing

    # save intermediate images at each step for inference
    images_each_step = ifelse(save_each_step, [initial_noise], nothing)

    for step in 1:diffusion_steps
        noisy_images = next_noisy_images

        # We start t = 1, and gradually decreases to t=0
        diffusion_times = ones(T, 1, 1, 1, num_images) .- step_size * step |> dev

        noise_rates, signal_rates = diffusion_schedules(
            diffusion_times,
            ddim.min_signal_rate,
            ddim.max_signal_rate
        )

        (pred_noises, pred_images), _ = denoise(
            ddim, 
            noisy_images, 
            noise_rates,
            signal_rates, 
            ps, 
            st
        )

        next_diffusion_times = diffusion_times .- step_size
        next_noise_rates, next_signal_rates = diffusion_schedules(
            next_diffusion_times,
            ddim.min_signal_rate,
            ddim.max_signal_rate
        )

        # see Eq. (12) in 2010.02502 with sigma=0
        next_noisy_images = next_signal_rates .* pred_images + next_noise_rates .* pred_noises

        if save_each_step
            push!(images_each_step, pred_images)
        end
    end

    return pred_images, images_each_step
end

"""
    denormalize

Denormalize the images.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function denormalize(
    ddim::DenoisingDiffusionImplicitModel{T}, 
    x::AbstractArray{T, 4},
    st
) where {T <: AbstractFloat}

    num_channels = size(x)[3]

    mean = reshape(st.running_mean, 1, 1, num_channels, 1)
    var = reshape(st.running_var, 1, 1, num_channels, 1)
    std = sqrt.(var .+ ddim.batchnorm.epsilon)

    return std .* x .+ mean
end

"""
    generate

Generate images using the DenoisingDiffusionImplicitModel.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function generate(
    ddim::DenoisingDiffusionImplicitModel{T}, 
    rng::AbstractRNG,
    image_shape::Tuple{Int, Int, Int, Int}, 
    diffusion_steps::Int, 
    ps,
    st::NamedTuple; 
    save_each_step=false,
    dev = gpu
) where {T}

    initial_noise = randn(rng, T, image_shape...) |> dev
    generated_images, images_each_step = reverse_diffusion(
        ddim, 
        initial_noise,
        diffusion_steps, 
        ps, 
        st;
        save_each_step=save_each_step,
        dev=dev
    )

    generated_images = denormalize(ddim, generated_images, st.batchnorm)
    clamp!(generated_images, 0.0f0, 1.0f0)

    if !isnothing(images_each_step)
        for (i, images) in enumerate(images_each_step)
            images_each_step[i] = denormalize(ddim, images, st.batchnorm)
            clamp!(images_each_step[i], 0.0f0, 1.0f0)
        end
    end

    return generated_images, images_each_step
end