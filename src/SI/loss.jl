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
    interpolant::Interpolant, 
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    loss = 0.0
    batch_size = size(x_0)[end]

    t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))
    
    z = randn!(rng, similar(x_1, size(x_1)))
    z = sqrt.(t) .* z

    g = interpolant.gamma(t) |> dev
    dg_dt = interpolant.dgamma_dt(t) |> dev

    x_history = x_0
    x_0 = x_0[:, :, :, end, :]

    I = interpolant.interpolant(x_0, x_1, t) .|> dev
    dI_dt = interpolant.dinterpolant_dt(x_0, x_1, t) .|> dev

    I = I .+ g .* z

    R = dI_dt .+ dg_dt .* z

    pred, st = velocity((I, x_history, pars, t), ps, st)

    loss = mean((pred - R).^2)

    # loss = loss + mean(pred.^2 - 2 .* pred .* R)

    return loss, st
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
    interpolant::Interpolant, 
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
    interpolant::Interpolant, 
    gamma::Gamma,
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
    interpolant::Interpolant, 
    ps::NamedTuple, 
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)
    batch_size = size(x_0)[end]

    t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))
    
    x_history = x_0
    x_0 = x_0[:, :, :, end, :]

    physics_vel_t = physics_velocity((x_history, x_history, x_history, x_history))
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

    I = (alpha_t + beta_t) .* x_0 .+ beta_t .* (physics_vel_t + physics_discrepancy) .+ g .* z
    R = (dalpha_dt + dbeta_dt) .* x_0 .+ dbeta_dt .* (physics_vel_t + physics_discrepancy) .+ dg_dt .* z

    model_vel_t, st = model_velocity((I, x_history, pars, t), ps, st)

    full_vel_t = model_vel_t + physics_vel_t

    loss = mean((full_vel_t - R).^2)

    return loss, st
end


"""
    get_action_matching_loss(
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
# function get_action_matching_loss(
#     x_0::AbstractArray, 
#     x_1::AbstractArray,
#     pars::AbstractArray,
#     action::Lux.AbstractExplicitLayer,
#     interpolant::Interpolant,
#     diffusion_coefficient::Function,
#     ps::NamedTuple, 
#     st::NamedTuple,
#     rng::AbstractRNG,
#     dev=gpu_device()
# )

#     batch_size = size(x_0)[end]

#     # Sample time
#     t_0 = zeros(Float32, (1, 1, 1, batch_size)) |> dev
#     t_1 = ones(Float32, (1, 1, 1, batch_size)) |> dev
#     t = rand!(rng, similar(x_1, 1, 1, 1, batch_size))

#     # Sample noise
#     # z = randn!(rng, similar(x_1, size(x_1)))
#     # z = diffusion_coefficient .* z

#     x_history = x_0
#     x_0 = x_0[:, :, :, end, :]

#     # Compute boundary terms
#     left_boundary = action((x_0, x_history, pars, t_0), ps, st)
#     right_boundary = action((x_1, x_history, pars, t_1), ps, st)

#     loss = left_boundary - right_boundary

#     # Compute time terms 
#     s_t_samples = interpolant.interpolant(x_0, x_1, t) .|> dev

#     forward_func_x = x -> action((x, x_history, pars, t), ps, st)
#     forward_func_t = t -> action((s_t_samples, x_history, pars, t), ps, st)

#     grad_s_t_term = ForwardDiff.gradient(forward_func_x, s_t_samples)
#     grad_s_t_term = sum(0.5 .* grad_s_t_term.^2, (1, 2))

#     ds_t_dt = ForwardDiff.derivative(forward_func_t, t)

#     0.5*diffusion_coefficient(t)**2*(jvp_val*eps).sum(1, keepdims=True)


#     funcres(x) = first(phi(x,res.minimizer))
#     dxu        = ForwardDiff.derivative.(funcres, Array(x_plot))
#     display(plot(x_plot,dxu,title = "Derivative",linewidth=3))







    

#     I = I .+ g .* z

#     R = dI_dt .+ dg_dt .* z

#     pred, st = velocity((I, x_history, pars, t), ps, st)

#     loss = mean((pred - R).^2)

#     # loss = loss + mean(pred.^2 - 2 .* pred .* R)

#     return loss, st
# end
