using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA


function get_loss(
    x_0, 
    t, 
    noise_scheduling, 
    model, 
    ps, 
    st, 
    rng,
    dev=gpu
)

    x_noisy, noise = forward_diffusion_sample(x_0, t, rng, noise_scheduling, dev)


    if length(size(t)) == 1
        t = reshape(t, (1, 1, 1, size(t)[1]))
    end

    noise_pred, st = model((x_noisy, t), ps, st)
    return mean(abs.(noise .- noise_pred).^2), st
end



