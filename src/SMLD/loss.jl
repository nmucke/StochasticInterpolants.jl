using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA


"""
    get_loss(
        x_0::AbstractArray, 
        t::AbstractArray, 
        model::ScoreMatchingLangevinDynamics, 
        marginal_prob_std, 
        eps=1e-5,
        dev=gpu_device()
    )

Computes the loss of the SMLD model
"""
function get_loss(
    x_0::AbstractArray, 
    t::AbstractArray, 
    model::ScoreMatchingLangevinDynamics, 
    ps::NamedTuple, 
    st::NamedTuple, 
    rng::AbstractRNG,
    dev=gpu_device()
)

    t = t  .* (1 .- model.eps) .+ model.eps  

    if length(size(t)) == 1
        t = reshape(t, (1, 1, 1, size(t)[1]))
    end

    z = randn(rng, size(x_0)) |> dev

    std_ = model.marginal_probability_std(t) |> dev

    perturbed_x_0 = x_0 .+ z .* std_

    score, st = model.unet((perturbed_x_0, t), ps, st)

    return mean((score .* std_ .+ z).^2), st

end