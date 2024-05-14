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
function euler_maruyama_sampler(
    model::UNet,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    marginal_probability_std::Function,
    diffusion_coefficient::Function,
    num_samples::Int,
    num_steps::Int,
    eps::AbstractFloat,
    dev=gpu
)
    t = ones((1, 1, 1, num_samples)) |> dev
    x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples) |> dev
    x = x .* Float32.(marginal_probability_std(t))
    timesteps = LinRange(1, eps, num_steps)# |> dev
    step_size = Float32.(timesteps[1] - timesteps[2]) |> dev

    for t in timesteps
        batch_time_step = ones((1, 1, 1, num_samples)) * t |> dev

        g = Float32.(diffusion_coefficient(batch_time_step))

        score, st = model((x, batch_time_step), ps, st)

        mean_x = x + (g.^2) .* score .* step_size

        z = randn(rng, size(x)) |> dev

        x = mean_x .+ sqrt.(step_size) .* g .* z  
    end

    return x
end



function sde_sampler(
    model::UNet,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    marginal_probability_std::Function,
    diffusion_coefficient::Function,
    num_samples::Int,
    num_steps::Int,
    eps::AbstractFloat,
    dev=gpu
)

            
    #x = model.sample(num_samples, ps, st_, rng, dev)
    x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples)
    x = x .* Float32.(marginal_probability_std(1.0)) |> dev
    t_span = (eps, 1.0)

    timesteps = LinRange(1, eps, num_steps)# |> dev
    dt = Float32.(timesteps[1] - timesteps[2]) |> dev


    stateful_score_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st)

    dudt(u, p, t) = -diffusion_coefficient(t).^2 .* stateful_score_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p) 
    g(u, p, t) = diffusion_coefficient(t) .* ones(size(u)) |> dev

    ff = SDEFunction{false}(dudt, g)
    prob = SDEProblem{false}(ff, g, x, reverse(t_span), ps)

    x = solve(
        prob,
        SRIW1(),
        dt = -dt,
        save_everystep = false,
        adaptive = false
    )
    x = x[:, :, :, :, end]

    return x

end


function ode_sampler(
    model::UNet,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    marginal_probability_std::Function,
    diffusion_coefficient::Function,
    num_samples::Int,
    num_steps::Int,
    eps::AbstractFloat,
    dev=gpu
)

            
    #x = model.sample(num_samples, ps, st_, rng, dev)
    x = randn(rng, Float32, model.upsample.size..., model.conv_in.in_chs, num_samples)
    x = x .* Float32.(marginal_probability_std(1.0)) |> dev
    t_span = (eps, 1.0)

    timesteps = LinRange(1, eps, num_steps)# |> dev
    dt = Float32.(timesteps[1] - timesteps[2]) |> dev

    stateful_score_NN = Lux.Experimental.StatefulLuxLayer(model, nothing, st)

    dudt(u, p, t) = -0.5*diffusion_coefficient(t).^2 .* stateful_score_NN((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p) 

    # ff = ODEFunction{false}(dudt, g)
    prob = ODEProblem{false}(dudt, x, reverse(t_span), ps)

    x = solve(
        prob,
        RK4(),
        dt = -dt,
        save_everystep = false, 
        adaptive = false
    )
    x = x[:, :, :, :, end]

    return x

end

