using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA


function get_loss(
    x_0::AbstractArray, 
    t::AbstractArray, 
    model::DenoisingDiffusionProbabilisticModel, 
    ps::NamedTuple, 
    st::NamedTuple, 
    rng::AbstractRNG,
    dev=gpu_device()
)

    x_noisy, noise = forward_diffusion_sample(x_0, t, rng, model.noise_scheduling, dev)


    if length(size(t)) == 1
        t = reshape(t, (1, 1, 1, size(t)[1]))
    end

    noise_pred, st = model.unet((x_noisy, t), ps, st)
    return mean(abs.(noise .- noise_pred).^2), st
end
