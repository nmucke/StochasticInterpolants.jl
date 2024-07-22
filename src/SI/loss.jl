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
        velocity::UNet,
        interpolant::Function, 
        gamma::Function,
        ps::NamedTuple, 
        st::NamedTuple,
        rng::AbstractRNG,
        dev=gpu_device()
    )

Computes the loss for the stochastic interpolant Model.
"""
function get_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    t::AbstractArray, 
    velocity::UNet,
    interpolant::Function, 
    gamma::Function,
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)

    z = randn(rng, size(x_0)) |> dev

    g, dg_dt = gamma(t, true) .|> dev

    I, dI_dt = interpolant(x_0, x_1, t, true) .|> dev

    I = I .+ g .* z

    velocity_pred, st_velocity = velocity((I, t), ps.velocity, st.velocity)

    st = (velocity=st_velocity, score=st.score)

    loss = mean(velocity_pred.^2 - 2.0 .* velocity_pred .* (dI_dt .+  dg_dt .* z))

    return loss, st
end


"""
    get_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        t::AbstractArray, 
        velocity::UNet,
        score::UNet,
        interpolant::Function, 
        gamma::Function,
        ps::NamedTuple, 
        st::NamedTuple,
        rng::AbstractRNG,
        dev=gpu_device()
    )

Computes the loss for the stochastic interpolant Model.
"""
function get_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    t::AbstractArray, 
    velocity::Lux.AbstractExplicitLayer,
    score::Lux.AbstractExplicitLayer,
    interpolant::Function, 
    gamma::Function,
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    num_steps = 1
    # t_vec = reshape(range(0f0, 1f0, num_steps), fill(1, ndims(x_0))..., :) |> dev
    #t_vec = rand(size(x_0)[end], num_steps) |> dev

    #loss = zeros(Float32, (1, 1, 1, size(x_0)[end])) |> dev

    batch_size = size(x_0)[end]

    st_score = st.score
    st_velocity = st.velocity

    velocity_loss = 0.0
    score_loss = 0.0
    # for i = 1:batch_size

    #     # copy along last dimension
    #     x_0_batch = repeat(x_0[:, :, :, i:i], 1, 1, 1, num_steps)
    #     x_1_batch = repeat(x_1[:, :, :, i:i], 1, 1, 1, num_steps)

    #     t = rand(rng, Float32, (1, 1, 1, num_steps)) |> dev 

    #     # z = randn(rng, size(x_0_batch)) |> dev
    #     # z = sqrt.(t) .* z


    #     z = randn(rng, size(x_0_batch)) |> dev
    #     z = sqrt.(t) .* z

    #     g, dg_dt = gamma(t, true) |> dev

    #     I, dI_dt = interpolant(x_0_batch, x_1_batch, t, true) .|> dev
        
    #     I = I .+ g .* z

    #     velocity_pred, st_velocity = velocity((I, t), ps.velocity, st_velocity)
    #     velocity_loss += mean(0.5*velocity_pred.^2 - velocity_pred .* (dI_dt + dg_dt .* z))

    #     score_pred, st_score = score((I, t), ps.score, st_score)
    #     #score_loss = mean(0.5*score_pred.^2 - 1 ./ g .* score_pred .* z)
    #     score_loss += mean(0.5*score_pred.^2 .- score_pred .* z)


    # end
    # velocity_loss = velocity_loss / batch_size
    # score_loss = score_loss / batch_size

    t = rand(rng, Float32, (1, 1, 1, batch_size)) |> dev 
    
    z = randn(rng, size(x_0)) |> dev

    I, dI_dt = interpolant(x_0, x_1, t, true) .|> dev
    
    g, dg_dt = gamma(t, true) |> dev
    I = I .+ g .* z

    velocity_pred, st_velocity = velocity((I, t), ps.velocity, st.velocity)
    velocity_loss = mean(0.5*velocity_pred.^2 - velocity_pred .* (dI_dt + dg_dt .* z))

    score_pred, st_score = score((I, t), ps.score, st.score)
    #score_loss = mean(0.5*score_pred.^2 - 1 ./ g .* score_pred .* z)
    score_loss = mean(0.5*score_pred.^2 .- score_pred .* z)


    st = (velocity=st_velocity, score=st_score)

    return velocity_loss + score_loss, st
end


"""
    get_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        t::AbstractArray, 
        velocity::UNet,
        score::UNet,
        interpolant::Function, 
        gamma::Function,
        ps::NamedTuple, 
        st::NamedTuple,
        rng::AbstractRNG,
        dev=gpu_device()
    )

Computes the loss for the stochastic interpolant Model.
"""
function get_forecasting_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    pars::AbstractArray,
    velocity::Lux.AbstractExplicitLayer,
    interpolant::Function, 
    gamma::Function,
    dgamma_dt::Function,
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    num_steps = 30
    loss = 0.0
    batch_size = size(x_0)[end]
    # for i = 1:batch_size

    #     # copy along last dimension
    #     x_0_batch = repeat(x_0[:, :, :, i:i], 1, 1, 1, num_steps)
    #     x_1_batch = repeat(x_1[:, :, :, i:i], 1, 1, 1, num_steps)

    #     t = rand(rng, Float32, (1, 1, 1, num_steps)) |> dev 

    #     z = randn(rng, size(x_0_batch)) |> dev
    #     z = sqrt.(t) .* z

    #     g = gamma(t) |> dev
    #     dg_dt = dgamma_dt(t) |> dev

    #     I, dI_dt = interpolant(x_0_batch, x_1_batch, t) .|> dev
    #     I = I .+ g .* z

    #     R = dI_dt .+ dg_dt .* z

    #     pred, st = velocity((I, x_0_batch, t), ps, st)

    #     loss = loss + mean((pred - R).^2)

    #     # loss = loss + mean(pred.^2 - 2 .* pred .* R)

    # end
    # loss = loss / batch_size

    # copy along last dimension

    # t = rand(rng, Float32, (1, 1, 1, batch_size)) |> dev 
    t = rand!(rng, similar(x_0, 1, 1, 1, batch_size))
    

    # z = randn(rng, size(x_0)) |> dev
    # z = sqrt.(t) .* z

    z = randn!(rng, similar(x_0, size(x_0)))
    z = sqrt.(t) .* z

    g = gamma(t) |> dev
    dg_dt = dgamma_dt(t) |> dev

    I, dI_dt = interpolant(x_0, x_1, t) .|> dev

    I = I .+ g .* z

    R = dI_dt .+ dg_dt .* z

    pred, st = velocity((I, x_0, pars, t), ps, st)


    loss = mean((pred - R).^2)



    # loss = loss + mean(pred.^2 - 2 .* pred .* R)

    return loss, st
end
