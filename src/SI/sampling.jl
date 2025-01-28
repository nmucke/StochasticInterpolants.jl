using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA
using Plots


# function sample_sde(
#     x_0::Int,
#     velocity::UNet,
#     score::UNet,
#     diffusion_coefficient::Function,
#     gamma::Function,
#     ps::NamedTuple,
#     st::NamedTuple,
#     num_steps::Int,
#     dev=gpu
# )

#     eps = 1e-5

#     # define time span
#     t_span = (eps, 1.0-eps)
#     timesteps = LinRange(t_span[1], t_span[2], num_steps)# |> dev
#     dt = Float32.(timesteps[2] - timesteps[1]) |> dev

#     # define the drift and diffusion functions
#     stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(velocity, nothing, st.velocity)
#     velocitydt(u, p, t) = stateful_velocity_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

#     stateful_score_NN = Lux.Experimental.StatefulLuxLayer(score, nothing, st.score)
#     scoredt(u, p, t) = diffusion_coefficient(t)/gamma(t) .* stateful_score_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

#     full_drift(u, p, t) = velocitydt(u, p.velocity, t) - scoredt(u, p.score, t)

#     g(u, p, t) = sqrt.(2 .* diffusion_coefficient(t)) .* ones(size(u)) |> dev

#     # define the SDE problem
#     ff = SDEFunction{false}(full_drift, g)
#     prob = SDEProblem{false}(ff, g, x_0, t_span, ps)

#     # solve the SDE
#     x = solve(
#         prob,
#         SRIW1(),
#         dt = dt,
#         save_everystep = false,
#         adaptive = false
#     )
#     x = x[:, :, :, :, end]

#     return x
# end

function sde_sampler(
    x_0::AbstractArray,
    velocity::Lux.AbstractExplicitLayer,
    score::Lux.AbstractExplicitLayer,
    diffusion_coefficient::Function,
    gamma::Function,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu_device()
)

    eps = 1e-5

    # define time span
    t_span = (eps, 1.0-eps)
    timesteps = LinRange(t_span[1], t_span[2], num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    # define the drift and diffusion functions
    stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(velocity, nothing, st.velocity)
    velocitydt(u, p, t) = stateful_velocity_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    stateful_score_NN = Lux.Experimental.StatefulLuxLayer(score, nothing, st.score)
    scoredt(u, p, t) = diffusion_coefficient(t)/gamma(t) .* stateful_score_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    full_drift(u, p, t) = velocitydt(u, p.velocity, t) - scoredt(u, p.score, t)

    g(u, p, t) = sqrt.(2 .* diffusion_coefficient(t)) .* ones(size(u)) |> dev

    # define the SDE problem
    ff = SDEFunction{false}(full_drift, g)
    prob = SDEProblem{false}(ff, g, x_0, t_span, ps)

    # solve the SDE
    x = solve(
        prob,
        SRIW1(),
        dt = dt,
        save_everystep = false,
        adaptive = false
    )
    x = x[:, :, :, :, end]

    return x

end

function sde_sampler(
    num_samples::Int,
    velocity::Lux.AbstractExplicitLayer,
    score::Lux.AbstractExplicitLayer,
    diffusion_coefficient::Function,
    gamma::Function,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    num_steps::Int,
    dev=gpu_device()
)

    # sample initial conditions
    # x = randn(rng, Float32, velocity.upsample.size..., velocity.conv_in.in_chs, num_samples) |> dev
    x = randn(rng, Float32, 32, 32, 2, num_samples) |> dev

    # sample from the SDE
    x = sde_sampler(
        x,
        velocity,
        score,
        diffusion_coefficient,
        gamma,
        ps,
        st,
        num_steps,
        dev
    )

    return x
end


# function sample_ode(
#     x_0::AbstractArray,
#     model::UNet,
#     ps::NamedTuple,
#     st::NamedTuple,
#     num_steps::Int,
#     dev=gpu
# )

#     t_span = (0.0, 1.0)

#     timesteps = LinRange(0, 1, num_steps)# |> dev
#     dt = Float32.(timesteps[2] - timesteps[1]) |> dev

#     stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st.velocity)

#     dudt(u, p, t) = stateful_velocity_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

#     # ff = ODEFunction{false}(dudt, g)
#     prob = ODEProblem(dudt, x_0, t_span, ps.velocity)

#     x = solve(
#         prob,
#         Tsit5(),
#         dt = dt,
#         save_everystep = false, 
#         adaptive = true
#     )
#     x = x[:, :, :, :, end]

#     return x

# end


function ode_sampler(
    x_0::AbstractArray,
    model,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu_device()
)

    t_span = (0.0, 1.0)

    timesteps = LinRange(0, 1, num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st)

    dudt(u, p, t) = stateful_velocity_NN((u, x_0, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    # ff = ODEFunction{false}(dudt, g)
    prob = ODEProblem(dudt, x_0, t_span, ps)

    x = solve(
        prob,
        Tsit5(),
        dt = dt,
        save_everystep = false, 
        adaptive = false
    )
    x = x[:, :, :, :, end]

    return x

end


function ode_sampler(
    num_samples::Int,
    model::Lux.AbstractExplicitLayer,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    num_steps::Int,
    dev=gpu_device()
)

    x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples) |> dev

    x = ode_sampler(
        x,
        model,
        ps,
        st,
        num_steps,
        dev
    )

    return x

end


# function drift_term(model, t, x, x_0, ps, st)

#     vel_t, st = model.velocity((x, x_0, t), ps, st)

#     term_1 = (1 .+ 1 ./(2 .- t)) .* vel_t

#     t = repeat(t, size(x_0)[1:3]...)

#     term_2 = (1f0 ./ (t .* (2f0 .- t))) .* (2f0 .* x .- (2f0 .- t) .* x_0)

#     drift_t = term_1 .- term_2
#     return drift_t, st
# end
function SDE_heun(
    drift_term,
    diffusion_term,
    x,
    pars,
    timesteps,
    ps::NamedTuple,
    st::NamedTuple,
    rng,
    dev=gpu_device()
)

    num_steps = length(timesteps)
    # (; z, dW) = cache

    z = similar(x, size(x))
    dW = similar(x, size(x))
    for i = 1:(num_steps-1)

        dt = Float32(timesteps[i+1] - timesteps[i]) |> dev

        # t_drift = timesteps[i] .* ones(Float32, 1, 1, 1, size(x)[end]) |> dev
        t = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])
        # t_diffusion = t_drift#timesteps[i:i]#repeat(t_drift, size(x)[1:3]...)   

        # z = randn(rng, size(x)) |> dev
        z = randn!(rng, z)
        dW .= sqrt(dt) .* z
        
        # Predictor from Euler step
        predictor_drift, st = drift_term(t, x, pars, ps, st)
        predictor_diffusion = diffusion_term(t, x, pars, ps, st)
        predictor = x .+ predictor_drift .* dt .+ predictor_diffusion .* dW

        # Corrector from Heun step
        heun_drift, st = drift_term(t .+ dt, predictor, pars, ps, st)
        drift_corrector = 0.5f0 * (predictor_drift .+ heun_drift)
        diffusion_corrector = 0.5f0 * (diffusion_term(t, x, pars, ps, st) .+ diffusion_term(t .+ dt, predictor, pars, ps, st))

        x .= x .+ drift_corrector .* dt + diffusion_corrector .* dW

    end

    return x
end

function SDE_euler_maruyama(
    drift_term::Function,
    diffusion_term::Function,
    x::AbstractArray,
    pars::AbstractArray,
    timesteps::AbstractArray,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)

    num_steps = length(timesteps)

    for i = 1:(num_steps-1)
        dt = Float32(timesteps[i+1] - timesteps[i]) |> dev

        # t_drift = timesteps[i] .* ones(Float32, 1, 1, 1, size(x)[end]) |> dev
        t = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])
        # t_diffusion = t_drift#timesteps[i:i]#repeat(t_drift, size(x)[1:3]...)   

        # z = randn(rng, size(x)) |> dev
        z = randn!(rng, similar(x, size(x)))
        dW = sqrt(dt) .* z
        
        # Predictor from Euler step
        drift, st = drift_term(t, x, pars, ps, st)
        diffusion = diffusion_term(t, x, pars, ps, st) 

        x = x .+ drift .* dt + diffusion .* dW

    end

    return x
end


function SDE_runge_kutta(
    drift_term::Function,
    diffusion_term::Function,
    x::AbstractArray,
    pars::AbstractArray,
    timesteps::AbstractArray,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device()
)

    num_steps = length(timesteps)
    dt = Float32(timesteps[2] - timesteps[1]) |> dev

    for i = 1:(num_steps-1)


        # t_drift = timesteps[i] .* ones(Float32, 1, 1, 1, size(x)[end]) |> dev
        t_drift = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])
        t_diffusion = timestep[i]#repeat(t_drift, size(x)[1:3]...)   

        # z = randn(rng, size(x)) |> dev
        z = randn!(rng, similar(x, size(x)))
        dW = sqrt(dt) .* z

        X0 = x
        t0_drift, t0_diffusion = t_drift, t_diffusion
        F0, st = drift_term(t0_drift, X0, pars, ps, st)
        G0 = diffusion_term(t0_diffusion, X0, pars, ps, st)

        X1 = x .+ 0.5f0 .* F0 .* dt .+ 0.5f0 .*  G0 .* dW
        t1_drift, t1_diffusion = t_drift .+ 0.5f0 .* dt, t_diffusion .+ 0.5f0 .* dt
        F1, st = drift_term(t1_drift, X1, pars, ps, st)
        G1 = diffusion_term(t1_diffusion, X1, pars, ps, st)

        X2 = x .+ 0.5f0 .* F1 .* dt .+ 0.5f0 .* G1 .* dW
        t2_drift, t2_diffusion = t_drift .+ 0.5f0 .* dt, t_diffusion .+ 0.5f0 .* dt
        F2, st = drift_term(t2_drift, X2, pars, ps, st)
        G2 = diffusion_term(t2_diffusion, X2, pars, ps, st)

        X3 = x .+ F2 .* dt .+ 0.5f0 .* G2 .* dW
        t3_drift, t3_diffusion = t_drift .+ dt, t_diffusion .+ dt
        F3, st = drift_term(t3_drift, X3, pars, ps, st)
        G3 = diffusion_term(t3_diffusion, X3, pars, ps, st)

        drift_t = (F0 .+ 2f0 .* F1 .+ 2f0 .* F2 .+ F3) ./ 6f0 .* dt
        diffusion_t = (G0 .+ 2f0 .* G1 .+ 2f0 .* G2 .+ G3) ./ 6f0 .* dW

        x = x .+ drift_t .+ diffusion_t


    end

    return x
end

function forecasting_sde_sampler(
    x_0::AbstractArray,
    pars::AbstractArray,
    model,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    rng::AbstractRNG,
    dev=gpu_device()
)

    # define time span
    timesteps = LinRange(0.0f0, 1.0f0, num_steps)
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev


    if model.gaussian_base_distribution
        x = randn!(rng, similar(x_0, size(x_0[:, :, :, end, :])))
    else
        x = x_0[:, :, :, end, :]
    end

    if typeof(model) == EncoderFollmerStochasticInterpolant
        
        t = zeros(1, 1, 1, size(x_0)[end])|> dev 
        x, st_new = model.encoder((x, x_0, pars, t), ps.encoder, st.encoder)
        @set st.encoder = st_new

        ps = ps.velocity
        st = st.velocity

    end

    # # Initial step
    # z = randn!(rng, similar(x, size(x)))
    # dW = sqrt(dt) .* z


    # t = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[1]) |> dev 
    # vel_t, st = model.drift_term(t, x, x_0, pars, ps, st; ode_mode=true);

    # # Initial step
    # # x = x + vel_t .* dt .+ model.diffusion_coefficient(t) .* dW
    # x = x + vel_t .* dt .+ model.diffusion_term(t, x, x_0, pars, ps, st) .* dW

    # Remaining steps
    drift_term(t, x, pars, ps, st) = model.drift_term(t, x, x_0, pars, ps, st);#; phys_vel=phys_vel_t)
    diffusion_term(t, x, pars, ps, st) = model.diffusion_term(t, x, x_0, pars, ps, st)
    # x = SDE_runge_kutta(
    # x = SDE_euler_maruyama(
    x = SDE_heun(
        drift_term,
        diffusion_term,
        x,
        pars,
        timesteps[1:end],
        ps,
        st,
        rng,
        dev
    )

    # # Final step
    # z = randn(rng, size(x_0)) |> dev
    # z = randn!(rng, similar(x, size(x)))

    # dt = Float32.(timesteps[end] - timesteps[end-1])
    # dW = sqrt(dt) .* z
    
    # t = timesteps[end-1] .* ones(1, 1, 1, size(x)[end]) |> dev
    # # vel_t, st = model.drift_term(t, x, x_0, pars, ps, st; ode_mode=true); #, phys_vel=phys_vel_t)
    # vel_t, st = model.drift_term(t, x, x_0, pars, ps, st; ode_mode=false);

    # t = repeat(t, size(x)[1:3]...)
    # # x = x + vel_t .* dt .+ model.gamma(t) .* dw  
    # x = x + vel_t .* dt .+ model.diffusion_term(t, x, x_0, pars, ps, st) .* dW

    if !isnothing(model.projection)
        x = model.projection(x, dev)
    end
    
    return x
end

function forecasting_latent_sde_sampler(
    x_0::AbstractArray,
    pars::AbstractArray,
    model,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    rng::AbstractRNG,
    dev=gpu_device()
)

    x = x_0[:, :, :, end, :]

    # define time span
    timesteps = LinRange(0.0f0, 1.0f0, num_steps)
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    # Initial step
    z = randn!(rng, similar(x, size(x)))
    dW = sqrt(dt) .* z

    t = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[1]) |> dev 
    vel_t, st = model.drift_term(t, x, x_0, pars, ps, st; ode_mode=true);

    x = x + vel_t .* dt .+ model.diffusion_term(t, x, x_0, pars, ps, st) .* dW

    # Remaining steps
    drift_term(t, x, pars, ps, st) = model.drift_term(t, x, x_0, pars, ps, st);
    diffusion_term(t, x, pars, ps, st) = model.diffusion_term(t, x, x_0, pars, ps, st)
    x = SDE_heun(
        drift_term,
        diffusion_term,
        x,
        pars,
        timesteps[2:end],
        ps,
        st,
        rng,
        dev
    )

    if typeof(model) == LatentFollmerStochasticInterpolant
        x_hf = model.autoencoder.decode(x)
    end

    if !isnothing(model.projection)
        x = model.projection(x, dev)
    end

    return x_hf, x
end

function ODE_runge_kutta(
    drift_term::Function,
    x::AbstractArray,
    pars::AbstractArray,
    timesteps::AbstractArray,
    ps::NamedTuple,
    st::NamedTuple,
    dev=gpu_device()
)

    num_steps = length(timesteps)
    @allowscalar dt = Float32(timesteps[2] - timesteps[1])

    for i = 1:(num_steps-1)

        # t_drift = fill!(similar(x, 1), timesteps[i])
        # t_drift = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])
        @allowscalar t_drift = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])

        X0 = x
        t0_drift = t_drift
        F0, st = drift_term(t0_drift, X0, pars, ps, st)

        X1 = x .+ 0.5f0 .* F0 .* dt
        t1_drift = t_drift .+ 0.5f0 .* dt
        F1, st = drift_term(t1_drift, X1, pars, ps, st)

        X2 = x .+ 0.5f0 .* F1 .* dt
        t2_drift = t_drift .+ 0.5f0 .* dt
        F2, st = drift_term(t2_drift, X2, pars, ps, st)

        X3 = x .+ F2 .* dt
        t3_drift = t_drift .+ dt
        F3, st = drift_term(t3_drift, X3, pars, ps, st)

        drift_t = (F0 .+ 2 .* F1 .+ 2 .* F2 .+ F3) ./ 6 .* dt

        x = x .+ drift_t

    end

    return x
end

function ODE_heun(
    drift_term::Function,
    x::AbstractArray,
    pars::AbstractArray,
    timesteps::AbstractArray,
    ps::NamedTuple,
    st::NamedTuple,
    dev=gpu_device()
)

    num_steps = length(timesteps)
    @allowscalar dt = Float32(timesteps[2] - timesteps[1])

    for i = 1:(num_steps-1)

        # t_drift = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])
        # @allowscalar t_drift = fill!(similar(x, 1), timesteps[i])
        @allowscalar t_drift = fill!(similar(x, 1, 1, 1, size(x)[end]), timesteps[i])

        F1, st = drift_term(t_drift, x, pars, ps, st)        
        X1 = x .+ F1 .* dt

        t1_drift = t_drift .+ dt
        F2, st = drift_term(t1_drift, X1, pars, ps, st)

        x = x .+ 0.5f0 .* (F1 .+ F2) .* dt

    end

    return x
end


function forecasting_ode_sampler(
    x_0::AbstractArray,
    pars::AbstractArray,
    model,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu_device()
)

    # define time span
    timesteps = LinRange(0.0f0, 1.0f0, num_steps) |> dev


    if typeof(model) == LatentFollmerStochasticInterpolant
        (H, W, C, len_history, batch_size) = size(x_0)
        (L_H, L_W, L_C) = model.autoencoder.latent_dimensions

        x_0 = reshape(x_0, H, W, C, len_history * batch_size)
        (x_0, _) = model.autoencoder.encode(x_0)
        x_0 = reshape(x_0, L_H, L_W, L_C, len_history, batch_size)
    end


    x_0 = x_0 |> dev

    # drift_term(t, x, pars, ps, st) = model.drift_term((x, x_0, pars, t), ps, st)


    drift_term(t, x, pars, ps, st) = model.drift_term(t, x, x_0, pars, ps, st; ode_mode=true)
    x = ODE_runge_kutta(
    # x = ODE_heun(
        drift_term,
        x_0[:, :, :, end, :],
        pars,
        timesteps,
        ps,
        st,
        dev
    )

    if !isnothing(model.projection)
        x = model.projection(x)
    end

    if typeof(model) == LatentFollmerStochasticInterpolant
        x = model.autoencoder.decode(x)
    end

    return x

end


# function forecasting_ode_sampler(
#     x_0::AbstractArray,
#     model::ForecastingStochasticInterpolant,
#     ps::NamedTuple,
#     st::NamedTuple,
#     num_steps::Int,
#     dev=gpu_device()
# )

#     t_span = (0.0, 1.0)

#     timesteps = LinRange(0, 1, num_steps)# |> dev
#     dt = Float32.(timesteps[2] - timesteps[1]) |> dev

#     stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(model.velocity, nothing, st)

#     dudt(u, p, t) = stateful_velocity_NN((u, x_0, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

#     # ff = ODEFunction{false}(dudt, g)
#     prob = ODEProblem(dudt, x_0, t_span, ps)

#     x = solve(
#         prob,
#         Tsit5(),
#         dt = dt,
#         save_everystep = false, 
#         adaptive = false
#     )
#     x = x[:, :, :, :, end]

#     return x

# end
