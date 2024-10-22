









function compute_multiple_SDE_steps(;
    init_condition::AbstractArray,
    parameters::AbstractArray,
    num_physical_steps::Int,
    num_generator_steps::Int,
    num_paths::Int,
    model,
    ps::NamedTuple,
    st::NamedTuple,
    rng::AbstractRNG,
    dev=gpu_device(),
    mask=nothing,
)
    cpu_dev = LuxCPUDevice()

    if !isnothing(mask)
        mask = mask |> dev 
    end
    
    
    sol = zeros(size(init_condition)[1:3]..., num_physical_steps, num_paths)

    sol[:, :, :, 1:model.velocity.len_history, :] = repeat(init_condition, 1, 1, 1, num_paths)

    pars = repeat(parameters, 1, num_paths) |> dev

    # x_n = sol[:, :, :, 1:model.velocity.len_history, :] |> dev
    for i in ProgressBar(model.velocity.len_history:(num_physical_steps - 1))

        # if !isnothing(mask)
        #     x_n = x_n .* mask
        # end

        
        x_n = forecasting_sde_sampler(
            sol[:, :, :, (i-model.velocity.len_history+1):i, :] |> dev, pars, model, ps, st, num_generator_steps, rng, dev
        )

        if !isnothing(mask)
            x_n = x_n .* mask
        end
        
        sol[:, :, :, i+1, :] = x_n |> cpu_dev

        if findmax(abs.(x_n))[1] > 1e6
            print("Trajectory diverged")
            break
        elseif any(isnan, x_n)
            print("NaN encountered")
            break
        end
    end

    return sol
end


function compute_multiple_ODE_steps(;
    init_condition::AbstractArray,
    parameters::AbstractArray,
    num_physical_steps::Int,
    num_generator_steps::Int,
    model,
    ps::NamedTuple,
    st::NamedTuple,
    dev=gpu_device(),
    mask=nothing,
)
    cpu_dev = LuxCPUDevice()

    num_trajectories = size(init_condition)[end]
    
    if !isnothing(mask)
        mask = mask |> dev 
    end
    
    sol = zeros(size(init_condition)[1:3]..., num_physical_steps, num_trajectories)

    sol[:, :, :, 1:model.velocity.len_history, :] = init_condition # reshape(init_condition, size(init_condition)..., num_trajectories)

    parameters = parameters |> dev

    # x_n = sol[:, :, :, 1, :] |> dev
    for i in ProgressBar(model.velocity.len_history:(num_physical_steps - 1))


        x_n = forecasting_ode_sampler(sol[:, :, :, (i-model.velocity.len_history+1):i, :] |> dev, parameters, model, ps, st, num_generator_steps, dev)
        
        if !isnothing(mask)
            x_n = x_n .* mask
        end

        sol[:, :, :, i+1, :] = x_n |> cpu_dev

        if findmax(abs.(x_n))[1] > 1e2
            print("Trajectory diverged")
            break
        elseif any(isnan, x_n)
            print("NaN encountered")
            break
        end
    end

    return sol
end
