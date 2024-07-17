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
using ProgressBars

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
        dev=gpu_device()
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
    dev=gpu_device()
)

    # if typeof(model) == ScoreMatchingLangevinDynamics
    #     get_t_batch(batch_size) = rand(rng, Float32, (batch_size,));

    # elseif typeof(model) == DenoisingDiffusionProbabilisticModel
    #     get_t_batch(batch_size) = rand(rng, 0:model.timesteps-1, (batch_size,));
    # end

    cpu_dev = LuxCPUDevice();


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
        dev=gpu_device()
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
    dev=gpu_device()
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
        
        (epoch % 10 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

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

            save_dir = joinpath("output/train/images/", @sprintf("sde_img_%i.pdf", epoch))
            savefig(plot(plot_list..., layout=(3,3)), save_dir)



            # x = model.ode_sample(num_samples, ps, st_, rng, dev)
                        
            # x = x |> cpu_dev
            # #x = x[:, :, 1, :]
            # x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2)

            # plot_list = []
            # for i in 1:9
            #     push!(plot_list, heatmap(x[:, :, i])); 
            # end

            # save_dir = joinpath("output/train/images/", @sprintf("ode_img_%i.pdf", epoch))
            # savefig(plot(plot_list..., layout=(3,3)), save_dir)

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
        dev=gpu_device()
    )

Train the Stochastic Interpolant model.
"""
function train_stochastic_interpolant(
    model::ForecastingStochasticInterpolant,
    ps::NamedTuple,
    st::NamedTuple,
    opt_state::NamedTuple,
    trainset_target_distribution::AbstractArray,
    num_epochs::Int,
    batch_size::Int,
    num_steps::Int,
    rng::AbstractRNG,
    trainset_init_distribution=nothing,
    trainset_pars_distribution=nothing,
    train_time_stepping=false,
    dev=gpu_device()
)

    min_lr = 1e-5
    max_lr = 1e-3 

    output_sde = true
    output_ode = false

    if train_time_stepping
        original_init_condition = trainset_init_distribution[:, :, :, 20*num_steps:21*num_steps]
        original_pars = trainset_pars_distribution[:, 1:1]
    end

    cpu_dev = LuxCPUDevice();

    num_target = size(trainset_target_distribution)[end]

    for epoch in 1:num_epochs
        running_loss = 0.0

        # Shuffle trainset
        train_ids = shuffle(rng, 1:num_target)
        trainset_target_distribution = trainset_target_distribution[:, :, :, train_ids]
        trainset_init_distribution = trainset_init_distribution[:, :, :, train_ids]
        trainset_pars_distribution = trainset_pars_distribution[:, train_ids]
    
        for i in 1:batch_size:size(trainset_target_distribution)[end]
            
            if i + batch_size - 1 > size(trainset_target_distribution)[end]
                break
            end

            x_1 = trainset_target_distribution[:, :, :, i:i+batch_size-1] |> dev
            x_0 = trainset_init_distribution[:, :, :, i:i+batch_size-1] |> dev
            pars = trainset_pars_distribution[:, i:i+batch_size-1] |> dev


            (loss, st), pb_f = Zygote.pullback(
                p -> model.loss(x_0, x_1, pars, p, st, rng, dev), ps
            );


            running_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)

            # remove all arrays from gpu_device
            
        end

        # Cosine annealing learning rate
        # opt_state = (; opt_state..., Adam(0.5 * (max_lr + min_lr) * (1 + cos(pi * epoch / num_epochs)), (0.9, 0.999), 1e-10))
        # new_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / num_epochs))
        # opt_state.rule = Adam(new_lr, (0.9, 0.999), 1e-8)
        
        running_loss /= floor(Int, size(trainset_target_distribution)[end] / batch_size)
        
        (epoch % 1 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

        if epoch % 5 == 0

            st_ = Lux.testmode(st)

            num_paths = 10

            if output_sde
                x = zeros(size(original_init_condition)..., num_paths)
                init_condition = reshape(original_init_condition, size(original_init_condition)..., 1)
                x[:, :, :, 1, :] = repeat(init_condition, 1, 1, 1, 1, num_paths)[:, :, :, 1, :]
                x_mean = mean(x[:, :, :, 1, :], dims=4)
                # x_mean = reshape(x_mean, size(x_mean)..., 1)

                pars = repeat(original_pars, 1, num_paths) |> dev


                error = []
                for i in ProgressBar(1:(num_steps - 1))

                    x_mean = repeat(x_mean, 1, 1, 1, num_paths)
                    
                    # pred = forecasting_sde_sampler(x[:, :, :, i, :] |> dev, model, ps, st_, 25, rng, dev)
                    pred = forecasting_sde_sampler(x[:, :, :, i, :] |> dev, pars, model, ps, st_, 25, rng, dev)
                    pred = pred |> cpu_dev
                    x[:, :, :, i+1, :] = pred
                    x_mean = mean(x[:, :, :, i+1, :], dims=4)

                    push!(error, mean((x_mean - original_init_condition[:, :, :, i+1] ).^2))

                    if findmax(abs.(x))[1] > 1e2
                        print("Trajectory diverged")
                        break
                    elseif any(isnan, x)
                        print("NaN encountered")
                        break
                    end
                end

                pred = nothing

                println("Time stepping error (SDE): ", mean(error))
                # x = sqrt.(x[:, :, 2, :, :].^2 + x[:, :, 3, :, :].^2)
                # x_true = sqrt.(original_init_condition[:, :, 2, :].^2 + original_init_condition[:, :, 3, :].^2)

                x = sqrt.(x[:, :, 1, :, :].^2 + x[:, :, 2, :, :].^2)
                x_true = sqrt.(original_init_condition[:, :, 1, :].^2 + original_init_condition[:, :, 2, :].^2)

                x_mean = mean(x, dims=4)
                x_std = std(x, dims=4)

                save_path = @sprintf("output/sde_SI_%i.gif", epoch)

                preds_to_save = (x_true, x_mean, x_mean-x_true, x_std, x[:, :, :, 1], x[:, :, :, 2], x[:, :, :, 3], x[:, :, :, 4])
                create_gif(preds_to_save, save_path, ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"])

                CUDA.reclaim()
                GC.gc()
            
            end

            if output_ode
                x = zeros(size(original_init_condition)...)
                x[:, :, :, 1] = original_init_condition[:, :, :, 1] |> cpu_dev

                error = []
                for i in 1:(num_steps - 1)
                    # init_condition = model.ode_sample(init_condition, ps, st_, dev)
                    # x[:, :, :, i] = init_condition |> cpu_dev

                    if i % 25 == 0
                        init_condition = original_init_condition[:, :, :, i:i] |> dev
                    else
                        init_condition = x[:, :, :, i:i] |> dev
                    end
                    pred = forecasting_ode_sampler(init_condition, model, ps, st_, 100, dev)

                    x[:, :, :, i+1] = pred |> cpu_dev
                    
                    push!(error, mean((x[:, :, :, i+1] - original_init_condition[:, :, :, i+1] ).^2))

                    if findmax(abs.(x))[1] > 1e2
                        break
                    end
                end


                println("Time stepping error (ODE): ", mean(error))

                x = sqrt.(x[:, :, 2, :].^2 + x[:, :, 3, :].^2)
                x_true = sqrt.(original_init_condition[:, :, 2, :].^2 + original_init_condition[:, :, 3, :].^2)

                save_path = @sprintf("output/ode_SI_%i.gif", epoch)
                create_gif(x, x_true, x-x_true, save_path)


                CUDA.reclaim()
                GC.gc()
            end
            

        end

    end

    return ps, st

end










