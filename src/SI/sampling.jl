using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA



"""
    Euler_Maruyama_sampler(
        model::ScoreMatchingLangevinDynamics,
        marginal_probability_std::Function,
        ps::NamedTuple,
        st::NamedTuple,
        rng::AbstractRNG,
        num_samples::Int,
        num_steps::Int,
        eps::Float32,
        dev=gpu
    )

    Samples from the SMLD model using the Euler-Maruyama method
"""
# function euler_maruyama_sampler(
#     model::UNet,
#     ps::NamedTuple,
#     st::NamedTuple,
#     rng::AbstractRNG,
#     marginal_probability_std::Function,
#     diffusion_coefficient::Function,
#     num_samples::Int,
#     num_steps::Int,
#     eps::AbstractFloat,
#     dev=gpu
# )
#     t = ones((1, 1, 1, num_samples)) |> dev
#     x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples) |> dev
#     x = x .* Float32.(marginal_probability_std(t))
#     timesteps = LinRange(1, eps, num_steps)# |> dev
#     step_size = Float32.(timesteps[1] - timesteps[2]) |> dev

#     for t in timesteps
#         batch_time_step = ones((1, 1, 1, num_samples)) * t |> dev

#         g = Float32.(diffusion_coefficient(batch_time_step))

#         score, st = model((x, batch_time_step), ps, st)

#         mean_x = x + (g.^2) .* score .* step_size

#         z = randn(rng, size(x)) |> dev

#         x = mean_x .+ sqrt.(step_size) .* g .* z  
#     end

#     return x
# end



function sde_sampler(
    num_samples::Int,
    drift::UNet,
    score::UNet,
    diffusion_coefficient::Function,
    gamma::Function,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    num_steps::Int,
    dev=gpu
)

    eps = 1e-5

    x = randn(rng, Float32, drift.upsample.size..., drift.conv_in.in_chs, num_samples) |> dev

    t_span = (eps, 1.0-eps)

    timesteps = LinRange(t_span[1], t_span[2], num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_drift_NN = Lux.Experimental.StatefulLuxLayer(drift, nothing, st.drift)
    driftdt(u, p, t) = stateful_drift_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    stateful_score_NN = Lux.Experimental.StatefulLuxLayer(score, nothing, st.score)
    scoredt(u, p, t) = diffusion_coefficient(t)/gamma(t) .* stateful_score_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    full_drift(u, p, t) = driftdt(u, p.drift, t) - scoredt(u, p.score, t)

    g(u, p, t) = sqrt.(2 .* diffusion_coefficient(t)) .* ones(size(u)) |> dev

    ff = SDEFunction{false}(full_drift, g)
    prob = SDEProblem{false}(ff, g, x, reverse(t_span), ps)

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
    x_0::AbstractArray,
    drift::UNet,
    score::UNet,
    diffusion_coefficient::Function,
    gamma::Function,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu
)

    t_span = (0.0, 1.0)

    timesteps = LinRange(0.0, 1.0, num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_drift_NN = Lux.Experimental.StatefulLuxLayer(drift, nothing, st.drift)
    driftdt(u, p, t) = stateful_drift_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    stateful_score_NN = Lux.Experimental.StatefulLuxLayer(score, nothing, st.score)
    scoredt(u, p, t) = diffusion_coefficient(t)/gamma(t) .* stateful_score_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

    full_drift(u, p, t) = driftdt(u, p.drift, t) - scoredt(u, p.score, t)
    
    g(u, p, t) = sqrt.(2 .* diffusion_coefficient(t)) .* ones(size(u)) |> dev

    ff = SDEFunction{false}(full_drift, g)
    prob = SDEProblem{false}(ff, g, x_0, reverse(t_span), ps)

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
    t_span = (0.0, 1.0)

    timesteps = LinRange(0, 1, num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_drift_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st.drift)

    dudt(u, p, t) = stateful_drift_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)
    
    # ff = ODEFunction{false}(dudt, g)
    prob = ODEProblem(dudt, x, t_span, ps.drift)

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
    x_0::AbstractArray,
    model::UNet,
    ps::NamedTuple,
    st::NamedTuple,
    num_steps::Int,
    dev=gpu
)

    t_span = (0.0, 1.0)

    timesteps = LinRange(0, 1, num_steps)# |> dev
    dt = Float32.(timesteps[2] - timesteps[1]) |> dev

    stateful_drift_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st)

    dudt(u, p, t) = stateful_drift_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p) 
    
    # ff = ODEFunction{false}(dudt, g)
    prob = ODEProblem(dudt, x_0, t_span, ps)

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

