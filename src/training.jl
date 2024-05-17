using Lux
using StochasticInterpolants
using Random
using CUDA
using NNlib
using Setfield
using MLDatasets
using Plots
using Optimisers
using Zygote
using LuxCUDA
using Images
using ImageFiltering
using Printf
using Statistics

"""
    train_diffusion_model(
        model,
        ps::NamedTuple,
        st::NamedTuple,
        opt_state::NamedTuple,
        trainset::AbstractArray,
        num_epochs::Int,
        batch_size::Int,
        num_samples::Int,
        rng::AbstractRNG,
        dev=gpu
    )

Train the Diffusion model.
"""
function train_diffusion_model(
    model,
    ps::NamedTuple,
    st::NamedTuple,
    opt_state::NamedTuple,
    trainset::AbstractArray,
    num_epochs::Int,
    batch_size::Int,
    num_samples::Int,
    rng::AbstractRNG,
    dev=gpu
)

    # if typeof(model) == ScoreMatchingLangevinDynamics
    #     get_t_batch(batch_size) = rand(rng, Float32, (batch_size,));

    # elseif typeof(model) == DenoisingDiffusionProbabilisticModel
    #     get_t_batch(batch_size) = rand(rng, 0:model.timesteps-1, (batch_size,));
    # end

    cpu_dev = LuxCPUDevice();

    loss = 

    num_train = size(trainset)[end] 
    for epoch in 1:num_epochs
        running_loss = 0.0

        # Shuffle trainset
        train_ids = shuffle(rng, 1:num_train)
        trainset = trainset[:, :, :, train_ids]
    
        for i in 1:batch_size:size(trainset)[end]
            
            if i + batch_size - 1 > size(trainset)[end]
                break
            end

            x = trainset[:, :, :, i:i+batch_size-1] |> dev;

            t = rand(rng, Float32, (batch_size,)) |> dev #get_t_batch(batch_size) |> dev;

            (loss, st), pb_f = Zygote.pullback(
                p -> get_loss(x, t, model, p, st, rng, dev), ps
            );
            running_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
            
            # GC.gc()

            if i % 1 == 0
                CUDA.reclaim()
            end
            

        end
        
        running_loss /= floor(Int, size(trainset)[end] / batch_size)
        
        (epoch % 10 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

        if epoch % 50 == 0

            st_ = Lux.testmode(st)
            x = model.sde_sample(num_samples, ps, st_, rng, dev)
                        
            x = x |> cpu_dev
            x = sqrt.(x[5:123, 5:123, 1, :].^2 + x[5:123, 5:123, 2, :].^2)   

            plot_list = []
            for i in 1:9
                push!(plot_list, heatmap(x[:, :, i])); 
            end

            save_dir = joinpath("output/train/images/", @sprintf("img_%i.pdf", epoch))
            savefig(plot(plot_list..., layout=(3,3)), save_dir)

        end

    end

    return ps, st

end

"""
    train_stochastic_interpolant(
        model::StochasticInterpolantModel,
        ps::NamedTuple,
        st::NamedTuple,
        opt_state::NamedTuple,
        trainset::AbstractArray,
        num_epochs::Int,
        batch_size::Int,
        num_samples::Int,
        rng::AbstractRNG,
        trainset_init_distribution::AbstractArray=nothing,
        init_and_target_are_correlated=false,
        train_time_stepping=false,
        dev=gpu
    )

Train the Stochastic Interpolant model.
"""
function train_stochastic_interpolant(
    model::StochasticInterpolantModel,
    ps::NamedTuple,
    st::NamedTuple,
    opt_state::NamedTuple,
    trainset_target_distribution::AbstractArray,
    num_epochs::Int,
    batch_size::Int,
    num_samples::Int,
    rng::AbstractRNG,
    dev=gpu
)

    cpu_dev = LuxCPUDevice();

    num_target = size(trainset_target_distribution)[end]
    for epoch in 1:num_epochs
        running_loss = 0.0

        trainset_init_distribution = randn(rng, Float32, size(trainset_target_distribution))

        target_ids = shuffle(rng, 1:num_target)
        trainset_target_distribution = trainset_target_distribution[:, :, :, target_ids]

        for i in 1:batch_size:size(trainset_target_distribution)[end]
            
            if i + batch_size - 1 > size(trainset_target_distribution)[end]
                break
            end

            x_1 = trainset_target_distribution[:, :, :, i:i+batch_size-1] |> dev;
            x_0 = trainset_init_distribution[:, :, :, i:i+batch_size-1] |> dev;

            t = rand(rng, Float32, (1, 1, 1, batch_size)) |> dev 

            (loss, st), pb_f = Zygote.pullback(
                p -> model.loss(x_0, x_1, t, p, st, rng, dev), ps
            );

            running_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
            

            if i % 1 == 0
                CUDA.reclaim()
            end
            

        end
        
        running_loss /= floor(Int, size(trainset_target_distribution)[end] / batch_size)
        
        (epoch % 25 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

        if epoch % 50 == 0

            st_ = Lux.testmode(st)

            x = model.sde_sample(num_samples, ps, st_, rng, dev)
                        
            x = x |> cpu_dev
            #x = x[:, :, 1, :]
            x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2)

            plot_list = []
            for i in 1:9
                push!(plot_list, heatmap(x[:, :, i])); 
            end

            save_dir = joinpath("output/train/images/", @sprintf("img_%i.pdf", epoch))
            savefig(plot(plot_list..., layout=(3,3)), save_dir)

        end

    end

    return ps, st

end




"""
    train_stochastic_interpolant(
        model::ConditionalStochasticInterpolant,
        ps::NamedTuple,
        st::NamedTuple,
        opt_state::NamedTuple,
        trainset::AbstractArray,
        num_epochs::Int,
        batch_size::Int,
        num_samples::Int,
        rng::AbstractRNG,
        trainset_init_distribution::AbstractArray=nothing,
        init_and_target_are_correlated=false,
        train_time_stepping=false,
        dev=gpu
    )

Train the Stochastic Interpolant model.
"""
function train_stochastic_interpolant(
    model::ConditionalStochasticInterpolant,
    ps::NamedTuple,
    st::NamedTuple,
    opt_state::NamedTuple,
    trainset_target_distribution::AbstractArray,
    num_epochs::Int,
    batch_size::Int,
    num_samples::Int,
    rng::AbstractRNG,
    trainset_init_distribution=nothing,
    init_and_target_are_correlated=false,
    train_time_stepping=false,
    dev=gpu
)

    init_distribution_is_provided = true

    if isnothing(trainset_init_distribution) 
        init_distribution_is_provided = false
    end

    if train_time_stepping
        original_init_condition = trainset_init_distribution[:, :, :, 1]
    end

    cpu_dev = LuxCPUDevice();

    num_target = size(trainset_target_distribution)[end]
    if init_distribution_is_provided
        num_init = size(trainset_init_distribution)[end]
    end
    for epoch in 1:num_epochs
        running_loss = 0.0

        # Shuffle trainset
        if init_and_target_are_correlated
            train_ids = shuffle(rng, 1:num_target)
            trainset_target_distribution = trainset_target_distribution[:, :, :, train_ids]
            trainset_init_distribution = trainset_init_distribution[:, :, :, train_ids]
        else

            if isnothing(trainset_init_distribution) 
                trainset_init_distribution = randn(rng, Float32, size(trainset_target_distribution))
                num_init = size(trainset_init_distribution)[end]
            end

            target_ids = shuffle(rng, 1:num_target)
            trainset_target_distribution = trainset_target_distribution[:, :, :, target_ids]

            init_ids = shuffle(rng, 1:num_init)
            trainset_init_distribution = trainset_init_distribution[:, :, :, init_ids]
        end
    
        for i in 1:batch_size:size(trainset_target_distribution)[end]
            
            if i + batch_size - 1 > size(trainset_target_distribution)[end]
                break
            end

            x_1 = trainset_target_distribution[:, :, :, i:i+batch_size-1] |> dev;
            x_0 = trainset_init_distribution[:, :, :, i:i+batch_size-1] |> dev;

            t = rand(rng, Float32, (1, 1, 1, batch_size)) |> dev 

            (loss, st), pb_f = Zygote.pullback(
                p -> get_loss(x_0, x_1, t, model, p, st, dev), ps
            );

            running_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
            

            if i % 1 == 0
                CUDA.reclaim()
            end
            

        end
        
        running_loss /= floor(Int, size(trainset_target_distribution)[end] / batch_size)
        
        (epoch % 25 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

        if epoch % 500 == 0

            st_ = Lux.testmode(st)

            if train_time_stepping
                init_condition = original_init_condition
                init_condition = reshape(init_condition, size(init_condition)[1], size(init_condition)[2], 2, 1)
                init_condition = init_condition |> dev
                

                num_steps = 200
                x = zeros(size(init_condition)[1:3]..., num_steps)
                x[:, :, :, 1] = init_condition[:, :, :, 1] |> cpu_dev
                for i in 2:num_steps
                    init_condition = model.ode_sample(num_samples, ps, st_, rng, init_condition, dev)
                    x[:, :, :, i] = init_condition |> cpu_dev

                    if findmax(abs.(x))[1] > 1e2
                        break
                    end
                end

                x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2)

                create_gif(x, "SI.gif")

            else

                if init_distribution_is_provided
                    x = model.ode_sample(num_samples, ps, st_, rng, dev, x_0=trainset_init_distribution[:, :, :, 1:num_samples])
                else
                    x = model.ode_sample(num_samples, ps, st_, rng, dev)
                end
                x = model.ode_sample(num_samples, ps, st_, rng, dev)
                            
                x = x |> cpu_dev
                #x = x[:, :, 1, :]
                x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2)

                plot_list = []
                for i in 1:9
                    push!(plot_list, heatmap(x[:, :, i])); 
                end

                save_dir = joinpath("output/train/images/", @sprintf("img_%i.pdf", epoch))
                savefig(plot(plot_list..., layout=(3,3)), save_dir)

            end

        end

    end

    return ps, st

end










