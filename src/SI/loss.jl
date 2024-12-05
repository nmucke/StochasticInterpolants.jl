using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA
using ForwardDiff

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
    velocity,
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
    get_forecasting_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        pars::AbstractArray,
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
function get_forecasting_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    pars::AbstractArray,
    velocity::Lux.AbstractExplicitLayer,
    interpolant::NamedTuple, 
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    loss = 0.0
    batch_size = size(x_0)[end]
    
    x_history = x_0
    x_0 = x_0[:, :, :, end, :]

    # for _ in 1:5
    t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))

    z = randn!(rng, similar(x_1, size(x_1)))
    z = sqrt.(t) .* z

    g = interpolant.gamma(t) |> dev
    dg_dt = interpolant.dgamma_dt(t) |> dev

    I = interpolant.interpolant(x_0, x_1, t) .|> dev
    dI_dt = interpolant.dinterpolant_dt(x_0, x_1, t) .|> dev

    I = I .+ g .* z

    R = dI_dt .+ dg_dt .* z

    pred, st = velocity((I, x_history, pars, t), ps, st)

    loss = loss + mean((pred - R).^2)

    return loss, st
end

"""
    get_forecasting_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        pars::AbstractArray,
        velocity::Lux.AbstractExplicitLayer,
        encoder::Lux.AbstractExplicitLayer,
        interpolant::Function, 
        gamma::Function,
        ps::NamedTuple, 
        st::NamedTuple,
        rng::AbstractRNG,
        dev=gpu_device()
    )

Computes the loss for the stochastic interpolant Model.
"""
function get_encoder_forecasting_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    pars::AbstractArray,
    velocity::Lux.AbstractExplicitLayer,
    encoder::Lux.AbstractExplicitLayer,
    interpolant::NamedTuple, 
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    loss = 0.0
    batch_size = size(x_0)[end]
    
    x_history = x_0
    x_0 = x_0[:, :, :, end, :]

    t = zeros(Float32, (1, 1, 1, batch_size)) |> dev

    encoder_pred, st_new = encoder((x_0, x_history, pars, t), ps.encoder, st.encoder)
    @set st.encoder = st_new

    enc_loss = mean((encoder_pred - x_1).^2)

    # energy_pred = sum(encoder_pred.^2, dims=(1, 2, 3))
    # energy_true = sum(x_1.^2, dims=(1, 2, 3))

    # energy_loss = mean((energy_pred - energy_true).^2)
    # energy_loss = energy_loss / mean(energy_true)

    t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))

    z = randn!(rng, similar(x_1, size(x_1)))
    z = sqrt.(t) .* z

    g = interpolant.gamma(t) |> dev
    dg_dt = interpolant.dgamma_dt(t) |> dev

    I = interpolant.interpolant(encoder_pred, x_1, t) .|> dev
    dI_dt = interpolant.dinterpolant_dt(encoder_pred, x_1, t) .|> dev

    I = I .+ g .* z

    R = dI_dt .+ dg_dt .* z

    pred, st_new = velocity((I, x_history, pars, t), ps.velocity, st.velocity)
    @set st.velocity = st_new

    SI_loss = mean((pred - R).^2)

    return enc_loss + SI_loss, st

    # Get all loss on the same scale

    # print("Encoder Loss: ", enc_loss, " Energy Loss: ", energy_loss, " SI Loss: ", SI_loss, "\n")

    # enc_loss = enc_loss / max_loss
    # energy_loss = energy_loss / max_loss
    # SI_loss = SI_loss / max_loss

    # return enc_loss + energy_loss + SI_loss, st
end


"""
    get_forecasting_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        pars::AbstractArray,
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
function get_forecasting_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    pars::AbstractArray,
    velocity::Lux.AbstractExplicitLayer,
    score::Lux.AbstractExplicitLayer,
    interpolant::NamedTuple, 
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    batch_size = size(x_0)[end]

    t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))
    
    z = randn!(rng, similar(x_1, size(x_1)))

    g = interpolant.gamma(t) |> dev
    dg_dt = interpolant.dgamma_dt(t) |> dev

    x_history = x_0
    x_0 = x_0[:, :, :, end, :]

    I = interpolant.interpolant(x_0, x_1, t) .|> dev
    dI_dt = interpolant.dinterpolant_dt(x_0, x_1, t) .|> dev

    I = I .+ g .* z

    R = dI_dt .+ dg_dt .* z

    velocity_pred, st_new = velocity((I, x_history, pars, t), ps.velocity, st.velocity)
    @set st.velocity = st_new
    velocity_loss = mean(velocity_pred.^2 - 2f0 .* R .* velocity_pred)

    score_pred, st_new = score((I, x_history, pars, t), ps.score, st.score)
    @set st.score = st_new
    score_loss = mean(score_pred.^2 - 2f0 .* z .* score_pred)

    return velocity_loss + score_loss, st
end


function get_forecasting_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    pars::AbstractArray,
    g_0::Lux.AbstractExplicitLayer, 
    g_1::Lux.AbstractExplicitLayer, 
    g_z::Lux.AbstractExplicitLayer,
    interpolant::NamedTuple, 
    gamma,
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    batch_size = size(x_0)[end]

    t = rand!(rng, similar(x_0, 1, 1, 1, batch_size))

    z = randn!(rng, similar(x_0, size(x_0)))

    g = gamma.gamma(t) |> dev

    I = interpolant.interpolant(x_0, x_1, t) .|> dev

    I = I .+ g .* z
    
    _g_0, st_new = g_0(I, pars, ps.g_0, st.g_0)
    @set st.g_0 = st_new

    _g_1, st_new = g_1(I, pars, ps.g_1, st.g_1)
    @set st.g_1 = st_new

    _g_z, st_new = g_z(I, pars, ps.g_z, st.g_z)
    @set st.g_z = st_new
    
    loss_g_0 = mean((_g_0 .^ 2 - 2 .* x_0 .* _g_0))
    loss_g_1 = mean((_g_1 .^ 2 - 2 .* x_1 .* _g_1))
    loss_g_z = mean((_g_z .^ 2 - 2 .* z .* _g_z))

    loss = loss_g_0 + loss_g_1 + loss_g_z

    return loss, st
end



"""
    get_physics_forecasting_loss(
        x_0::AbstractArray, 
        x_1::AbstractArray,
        pars::AbstractArray,
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
function get_physics_forecasting_loss(
    x_0::AbstractArray, 
    x_1::AbstractArray,
    pars::AbstractArray,
    model_velocity::Lux.AbstractExplicitLayer,
    physics_velocity::Function,
    interpolant::NamedTuple, 
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    batch_size = size(x_0)[end]

    t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))
    
    x_history = x_0
    x_0 = x_0[:, :, :, end, :]

    physics_vel_t = physics_velocity((x_0, x_history, pars, t))
    physics_pred = x_0 + physics_vel_t

    physics_discrepancy = physics_pred - x_1


    z = randn!(rng, similar(x_1, size(x_1)))
    z = sqrt.(t) .* z

    g = interpolant.gamma(t) |> dev
    dg_dt = interpolant.dgamma_dt(t) |> dev

    alpha_t = interpolant.alpha(t) |> dev
    dalpha_dt = interpolant.dalpha_dt(t) |> dev

    beta_t = interpolant.beta(t) |> dev
    dbeta_dt = interpolant.dbeta_dt(t) |> dev

    # I = (alpha_t + beta_t) .* x_0 .+ beta_t .* (physics_vel_t + physics_discrepancy) .+ g .* z
    # R = (dalpha_dt + dbeta_dt) .* x_0 .+ dbeta_dt .* (physics_vel_t + physics_discrepancy) .+ dg_dt .* z

    I = x_0 .+ beta_t .* physics_vel_t .+ beta_t .* physics_discrepancy .+ g .* z
    # R = dbeta_dt * physics_vel_t + dbeta_dt * physics_discrepancy + dg_dt .* z

    model_vel_t, st = model_velocity((I, x_history, pars, t), ps, st)

    loss = mean((model_vel_t - physics_discrepancy + dg_dt .* z).^2)

    # full_vel_t = model_vel_t + physics_vel_t

    # loss = mean((full_vel_t - R).^2)

    return loss, st
end

