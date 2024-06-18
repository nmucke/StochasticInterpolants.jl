using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA


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
    velocity::UNet,
    score::UNet,
    diffusion_coefficient::Function,
    gamma::Function,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu
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
    velocity::UNet,
    score::UNet,
    diffusion_coefficient::Function,
    gamma::Function,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    num_steps::Int,
    dev=gpu
)

    # sample initial conditions
    x = randn(rng, Float32, velocity.upsample.size..., velocity.conv_in.in_chs, num_samples) |> dev

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
    model::ConditionalUNet,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu
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
    model::UNet,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    num_steps::Int,
    dev=gpu
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


function drift_term(model, t, x, x_0, ps, st)

    vel_t, st = model.velocity((x, x_0, t), ps, st)

    term_1 = (1 .+ 1 ./(2 .- t)) .* vel_t


    t = repeat(t, size(x_0)[1:3]...)

    term_2 = (1f0 ./ (t .* (2f0 .- t))) .* (2f0 .* x .- (2f0 .- t) .* x_0)

    drift_t = term_1 .- term_2
    return drift_t
end


function forecasting_sde_sampler(
    x_0::AbstractArray,
    model::ForecastingStochasticInterpolant,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    rng::AbstractRNG,
    dev=gpu
)

    epsilon = 0.1f0

    # define time span
    t_span = (0.0, 1.0)
    timesteps = LinRange(t_span[1], t_span[2], num_steps) |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    z = randn(rng, size(x_0)) |> dev

    x = x_0

    z = randn(rng, size(x_0)) |> dev
    t = timesteps[1] .* ones(1, 1, 1, size(x_0)[end]) |> dev
    vel_t, st = model.velocity(((x, x_0, t)), ps, st)

    t = repeat(t, size(x_0)[1:3]...)
    x = x + vel_t .* dt .+ model.gamma(t) .* sqrt.(dt) .* z
    for i = 2:(num_steps-1)

        t = timesteps[i] .* ones(Float32, 1, 1, 1, size(x_0)[end]) |> dev

        z = randn(rng, size(x)) |> dev

        drift_t = drift_term(model, t, x, x_0, ps, st)

        t = repeat(t, size(x_0)[1:3]...)    
        diff_t = epsilon .* sqrt.((1 .- t) .* (3 .- t))

        x = x + drift_t .* dt + diff_t .* sqrt.(dt) .* z
    end

    return x

end

# function forecasting_sde_sampler(
#     x_0::AbstractArray,
#     model::ForecastingStochasticInterpolant,
#     ps::NamedTuple,
#     st::NamedTuple,
#     num_steps::Int,
#     rng::AbstractRNG,
#     dev=gpu
# )
#     # define time span
#     t_span = (0.0, 1.0)
#     timesteps = LinRange(t_span[1], t_span[2], num_steps) |> dev
#     dt = Float32.(timesteps[2] - timesteps[1]) |> dev


#     ddiff_coeff_dt(t) = (t .- 1f0) ./ sqrt.(t.^2 .- 4f0 .* t .+ 3f0) |> dev

#     # define the drift and diffusion functions
#     # stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(model.velocity, nothing, st);
#     # velocity(u, p, t) = stateful_velocity_NN((u, x_0, t .* ones(1, 1, 1, size(u)[end])) |> dev, p);

#     # score(u, p, t) = model.score((u, x_0, t .* ones(1, 1, 1, size(u)[end])) |> dev, p, st);

#     # full_drift(u, p, t) = velocity(u, p, t) + 0.5 .* (model.diffusion_coefficient(t).^2 - model.gamma(t).^2) .* score(u, p, t)    #(lol(t).^2 - lol(t).^2) .* scoredt(u, p, t)

#     # g(u, p, t) = model.diffusion_coefficient(t) .* ones(size(u)) |> dev;

#     z = randn(rng, size(x_0)) |> dev
#     # x = x_0 + velocity(x_0, ps, 0.0) * dt + model.gamma(0.0) .* sqrt.(dt) .* z

#     x = x_0

#     z = randn(rng, size(x_0)) |> dev
#     t = timesteps[1] .* ones(1, 1, 1, size(x_0)[end]) |> dev
#     vel_t, st = model.velocity(((x, x_0, t)), ps, st)

#     x = x + vel_t .* dt .+ model.gamma(t) .* sqrt.(dt) .* z# .+ 0.5f0 .* model.gamma(t) .* (-1f0) .* (dt .* z.^2 .- dt)
#     for i = 2:(num_steps-1)

#         t = timesteps[i] .* ones(Float32, 1, 1, 1, size(x_0)[end]) |> dev

#         z = randn(rng, size(x)) |> dev

#         vel_t, st = model.velocity((x, x_0, t), ps, st)

#         t = repeat(t, size(x_0)[1:3]...)

#         score_t = model.score((x, x_0, t), vel_t)

#         diff_coeff = model.diffusion_coefficient(t)
#         gamma = model.gamma(t)

#         drift_t = vel_t + 0.5f0 .* (diff_coeff.^2 - gamma.^2) .* score_t

#         x = x + drift_t .* dt + diff_coeff .* sqrt.(dt) .* z# + 0.5f0 .* diff_coeff .* ddiff_coeff_dt(t) .* (dt .* z.^2 .- dt)
#     end

#     # t = timesteps[num_steps-1] .* ones(Float32, 1, 1, 1, size(x_0)[end]) |> dev
#     # vel_t, st = model.velocity(((x, x_0, t)), ps, st)
#     # t = repeat(t, size(x_0)[1:3]...)
#     # score_t = model.score((x, x_0, t), vel_t)
#     # diff_coeff = model.diffusion_coefficient(t)
#     # gamma = model.gamma(t)
#     # drift_t = vel_t + 0.5f0 .* (diff_coeff.^2 - gamma.^2) .* score_t
#     # x = x + drift_t .* dt
#     # x = x + vel_t .* dt

#     # define the SDE problem
#     # ff = SDEFunction{false}(full_drift, g) |> dev;
#     # prob = SDEProblem{false}(ff, g, x_0, t_span, ps) |> dev;

#     # solve the SDE
#     # x = solve(
#     #     prob,
#     #     SRIW1(),
#     #     dt = dt,
#     #     save_everystep = false,
#     #     adaptive = false
#     # )

#     return x #x[:, :, :, :, end]

# end


function forecasting_ode_sampler(
    x_0::AbstractArray,
    model::ForecastingStochasticInterpolant,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu
)

    t_span = (0.0, 1.0)

    timesteps = LinRange(0, 1, num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(model.velocity, nothing, st)

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