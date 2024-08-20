using Lux
using StochasticInterpolants
using Random
using CUDA
using NNlib
using Setfield
# using MLDatasets
using Plots
using Optimisers
using Zygote
using LuxCUDA
# using Images
# using ImageFiltering
using Printf
using Statistics
using ProgressBars

"""
    train_stochastic_interpolant(
        model,
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
    model,
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
    normalize_data=nothing,
    mask=nothing,
    rng::AbstractRNG,
    dev=gpu_device()
)

    output_sde = true
    output_ode = true

    num_steps = size(testset, 4)


    cpu_dev = LuxCPUDevice();

    num_target = size(trainset_target_distribution)[end]

    best_loss = 1e8

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


        if epoch % 10 == 0

            CUDA.reclaim()
            GC.gc()

            st_ = Lux.testmode(st)


            if output_sde
                num_test_trajectories = size(testset)[end]
                num_channels = size(testset, 3)
                num_test_steps = size(testset, 4)
                
                if !isnothing(normalize_data)
                    x_true = normalize_data.inverse_transform(testset)
                else
                    x_true = testset
                end

                
                pathwise_MSE = 0
                mean_MSE = 0
                x = zeros(size(testset)[1:3]..., num_test_steps, num_test_paths)
                for i = 1:num_test_trajectories

                    test_init_condition = testset[:, :, :, 1:1, i]
                    test_pars = testset_pars[:, 1:1, i]

                    x = compute_multiple_SDE_steps(
                        init_condition=test_init_condition,
                        parameters=test_pars,
                        num_physical_steps=num_test_steps,
                        num_generator_steps=100,
                        num_paths=num_test_paths,
                        model=model,
                        ps=ps,
                        st=st_,
                        rng=rng,
                        dev=dev,
                        mask=mask,
                    )

                    if !isnothing(normalize_data)
                        x = normalize_data.inverse_transform(x)
                    end

                    if !isnothing(mask)
                        x = x .* mask
                    end

                    mean_pathwise_MSE, mean_mean_MSE = compute_RMSE(
                        x_true[:, :, :, :, i], x, mask,
                    )

                    pathwise_MSE += mean_pathwise_MSE
                    mean_MSE += mean_mean_MSE

                end

                true_freq, true_fft = compute_temporal_frequency(x_true[:, :, :, :, num_test_trajectories:num_test_trajectories])
                pred_freq, pred_fft = compute_temporal_frequency(x)
                pred_fft_mean = mean(pred_fft, dims=2)
                pred_fft_min = minimum(pred_fft, dims=2)
                pred_fft_max = maximum(pred_fft, dims=2)

                plot(log2.(true_freq), log10.(true_fft .* true_fft .* true_fft), color=:blue, label="True", linewidth=3)
                plot!(log2.(pred_freq), log10.(pred_fft_mean .* pred_fft_mean .* pred_fft_mean), color=:red, label="Pred", linewidth=3)
                plot!(log2.(pred_freq), log10.(pred_fft_min .* pred_fft_min .* pred_fft_min), linestyle=:dash, color=:red)
                plot!(log2.(pred_freq), log10.(pred_fft_max .* pred_fft_max .* pred_fft_max), linestyle=:dash, color=:red)
                
                frequency_save_path = @sprintf("output/sde_frequency_%i.png", epoch)
                savefig(frequency_save_path)


                pathwise_MSE /= num_test_trajectories
                mean_MSE /= num_test_trajectories

                println("Mean of pathwise MSE: ", pathwise_MSE)
                println("Mean of mean MSE (SDE): ", mean_MSE)

                x_mean = mean(x, dims=5)[:, :, :, :, 1];
                x_std = std(x, dims=5)[:, :, :, :, 1];

                gif_save_path = @sprintf("output/sde_SI_%i.gif", epoch)
                preds_to_save = (
                    x_true[:, :, 4, :, num_test_trajectories], 
                    x_mean[:, :, 4, :], 
                    Float16.(x_mean[:, :, 4, :]-x_true[:, :, 4, :, num_test_trajectories]), 
                    Float16.(x_std[:, :, 4, :]), 
                    x[:, :, 4, :, 1], 
                    x[:, :, 4, :, 2], 
                    x[:, :, 4, :, 3], 
                    x[:, :, 4, :, 4]
                );
                create_gif(
                    preds_to_save, 
                    gif_save_path, 
                    ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"]
                )

                if !isnothing(model_save_dir) && pathwise_MSE < best_loss
                    save_checkpoint(
                        ps=ps, 
                        st=st,
                        opt_st=opt_state,
                        output_dir=model_save_dir,
                        epoch=epoch
                    )

                    best_loss = pathwise_MSE
                end

                
                CUDA.reclaim()
                GC.gc()
            
            
            end

            pathwise_MSE = 0
            mean_MSE = 0
            if output_ode
                num_test_trajectories = size(testset)[end]
                num_channels = size(testset, 3)
                num_test_steps = size(testset, 4)
                
                if !isnothing(normalize_data)
                    x_true = normalize_data.inverse_transform(testset)
                else
                    x_true = testset
                end
                
                x = compute_multiple_ODE_steps(
                    init_condition=testset[:, :, :, 1, :],
                    parameters=testset_pars[:, 1, :],
                    num_physical_steps=num_test_steps,
                    num_generator_steps=100,
                    model=model,
                    ps=ps,
                    st=st_,
                    dev=dev,
                    mask=mask,
                )


                if !isnothing(normalize_data)
                    x = normalize_data.inverse_transform(x)
                end

                if !isnothing(mask)
                    x = x .* mask
                end

                
                true_freq, true_fft = compute_temporal_frequency(x_true[:, :, :, :, num_test_trajectories:num_test_trajectories])
                pred_freq, pred_fft = compute_temporal_frequency(x)
                pred_fft = pred_fft[:, num_test_trajectories]

                plot(log2.(true_freq), log10.(true_fft .* true_fft .* true_fft), color=:blue, label="True", linewidth=3)
                plot!(log2.(pred_freq), log10.(pred_fft .* pred_fft .* pred_fft_mean), color=:red, label="Pred", linewidth=3)
                
                frequency_save_path = @sprintf("output/ode_frequency_%i.png", epoch)
                savefig(frequency_save_path)



                num_channels = size(x, 3)

                for i = 1:num_test_trajectories
                    mean_pathwise_MSE, mean_mean_MSE = compute_RMSE(
                        testset[:, :, :, :, i:i], x[:, :, :, :, i], mask,
                    )

                    pathwise_MSE += mean_pathwise_MSE
                    mean_MSE += mean_mean_MSE

                end
                
                x = x[:, :, 4, :, 1]
                x_true = x_true[:, :, 4, :, 1]
                
                save_path = @sprintf("output/ode_SI_%i.gif", epoch)
                
                preds_to_save = (x_true, x, x-x_true)
                create_gif(preds_to_save, save_path, ["True", "Pred", "Error"])

                println("MSE (ODE): ", pathwise_MSE)
                
                CUDA.reclaim()
                GC.gc()

            end


        end

    end

    return ps, st

end








