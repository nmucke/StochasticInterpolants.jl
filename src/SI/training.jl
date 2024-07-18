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
function train_stochastic_interpolant(;
    model::ForecastingStochasticInterpolant,
    ps::NamedTuple,
    st::NamedTuple,
    opt_state::NamedTuple,
    trainset_target_distribution::AbstractArray,
    trainset_init_distribution::AbstractArray=nothing,
    trainset_pars_distribution::AbstractArray=nothing,
    testset::AbstractArray=nothing,
    testset_pars::AbstractArray=nothing,
    num_test_paths::Int = 5,
    model_save_dir::String=nothing,
    num_epochs::Int=1000,
    batch_size::Int=8,
    rng::AbstractRNG,
    dev=gpu_device()
)

    output_sde = true
    output_ode = true

    num_steps = size(testset, 4)

    test_init_condition = testset[:, :, :, 1, 1:1]
    test_pars = testset_pars[:, 1:1, 1]
    num_test_steps = size(testset, 4)

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
            
        end

        # Cosine annealing learning rate
        # opt_state = (; opt_state..., Adam(0.5 * (max_lr + min_lr) * (1 + cos(pi * epoch / num_epochs)), (0.9, 0.999), 1e-10))
        # new_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / num_epochs))
        # opt_state.rule = Adam(new_lr, (0.9, 0.999), 1e-8)
        
        running_loss /= floor(Int, size(trainset_target_distribution)[end] / batch_size)
        
        if epoch % 1 == 0
            print("Loss Value after $epoch iterations: $running_loss \n")
        end

        if epoch % 1 == 0

            if !isnothing(model_save_dir)
                save_checkpoint(
                    ps=ps, 
                    st=st,
                    opt_st=opt_state,
                    output_dir=model_save_dir,
                    epoch=epoch
                )
            end

            st_ = Lux.testmode(st)


            if output_sde

                x = compute_multiple_SDE_steps(
                    init_condition=test_init_condition,
                    parameters=test_pars,
                    num_physical_steps=num_test_steps,
                    num_generator_steps=25,
                    num_paths=num_test_paths,
                    model=model,
                    ps=ps,
                    st=st_,
                    rng=rng,
                    dev=dev
                )

                # println("Time stepping error (SDE): ", mean(error))

                MSE = 0.0f0
                for i = 1:num_test_paths
                    MSE += mean((x[:, :, 4, :, i] - testset[:, :, 4, :, 1]).^2)
                end

                MSE /= num_test_paths
                println("MSE over paths (SDE): ", MSE)

                x = x[:, :, 4, :, :]
                x_true = testset[:, :, 4, :, 1]

                x_mean = mean(x, dims=4)
                x_std = std(x, dims=4)

                MSE = mean((x_mean - x_true).^2)
                println("MSE over mean (SDE): ", MSE)

                save_path = @sprintf("output/sde_SI_%i.gif", epoch)

                preds_to_save = (x_true, x_mean, x_mean-x_true, x_std, x[:, :, :, 1], x[:, :, :, 2], x[:, :, :, 3], x[:, :, :, 4])
                create_gif(preds_to_save, save_path, ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"])

                CUDA.reclaim()
                GC.gc()
            
            end

            if output_ode
                
                x = compute_multiple_ODE_steps(
                    init_condition=test_init_condition,
                    parameters=test_pars,
                    num_physical_steps=num_test_steps,
                    num_generator_steps=25,
                    model=model,
                    ps=ps,
                    st=st_,
                    dev=dev
                )

                MSE = mean((x[:, :, 4, :] - testset[:, :, 4, :, 1]).^2)
                println("MSE (ODE): ", MSE)

                # println("Time stepping error (ODE): ", mean(error))

                x = x[:, :, 4, :]
                x_true = testset[:, :, 4, :, 1]


                save_path = @sprintf("output/ode_SI_%i.gif", epoch)

                preds_to_save = (x_true, x, x-x_true)
                create_gif(preds_to_save, save_path, ["True", "Pred", "Error"])

                CUDA.reclaim()
                GC.gc()
            end
            

        end

    end

    return ps, st

end








