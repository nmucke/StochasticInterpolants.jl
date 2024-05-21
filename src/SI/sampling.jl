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
    model::UNet,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu
)

    # x = sample_ode(
    #     x_0,
    #     model,
    #     ps,
    #     st,
    #     num_steps,
    #     dev
    # )

    t_span = (0.0, 1.0)

    timesteps = LinRange(0, 1, num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_velocity_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st.velocity)

    dudt(u, p, t) = stateful_velocity_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    # ff = ODEFunction{false}(dudt, g)
    prob = ODEProblem(dudt, x_0, t_span, ps.velocity)

    x = solve(
        prob,
        Tsit5(),
        dt = dt,
        save_everystep = false, 
        adaptive = true
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


