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
        x_1::AbstractArray,
        t::AbstractArray, 
        drift::UNet,
        interpolant::Function, 
        gamma::Function,
        ps::NamedTuple, 
        st::NamedTuple,
        rng::AbstractRNG,
        dev=gpu
    )

Computes the loss for the stochastic interpolant Model.
"""
function get_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    t::AbstractArray, 
    drift::UNet,
    interpolant::Function, 
    gamma::Function,
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu
)

    z = randn(rng, size(x_0)) |> dev

    g, dg_dt = gamma(t, true) .|> dev

    I, dI_dt = interpolant(x_0, x_1, t, true) .|> dev

    I = I .+ g .* z

    drift_pred, st_drift = drift((I, t), ps.drift, st.drift)

    st = (drift=st_drift, score=st.score)

    loss = mean(drift_pred.^2 - 2.0 .* drift_pred .* (dI_dt .+  dg_dt .* z))

    return loss, st
end


"""
    get_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        t::AbstractArray, 
        drift::UNet,
        score::UNet,
        interpolant::Function, 
        gamma::Function,
        ps::NamedTuple, 
        st::NamedTuple,
        rng::AbstractRNG,
        dev=gpu
    )

Computes the loss for the stochastic interpolant Model.
"""
function get_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    t::AbstractArray, 
    drift::UNet,
    score::UNet,
    interpolant::Function, 
    gamma::Function,
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu
)
    z = randn(rng, size(x_0)) |> dev

    g = gamma(t, false) |> dev

    I, dI_dt = interpolant(x_0, x_1, t, true) .|> dev
    
    I = I .+ g .* z

    drift_pred, st_drift = drift((I, t), ps.drift, st.drift)
    drift_loss = mean(0.5*drift_pred.^2 - drift_pred .* dI_dt)

    score_pred, st_score = score((I, t), ps.score, st.score)
    #score_loss = mean(0.5*score_pred.^2 - 1 ./ g .* score_pred .* z)
    score_loss = mean(0.5*score_pred.^2 .- score_pred .* z)

    st = (drift=st_drift, score=st_score)

    return drift_loss + score_loss, st
end
