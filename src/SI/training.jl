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
    cpu_dev = LuxCPUDevice();

    # If the model is a LatentFollmerStochasticInterpolant, we need to encode the data
    # before training, to train the SI in the latent space.
    if typeof(model) == LatentFollmerStochasticInterpolant
        (L_H, L_W, L_C) = model.autoencoder.latent_dimensions
        (H, W, C, len_history) = size(trainset_init_distribution)[1:4]

        latent_target_distribution = zeros(L_H, L_W, L_C, size(trainset_target_distribution)[end])
        for i in 1:batch_size:size(trainset_target_distribution)[end]
            bs = batch_size
            if i + bs - 1 > size(trainset_target_distribution)[end]
                bs = size(trainset_target_distribution)[end] - i + 1
            end
            x = trainset_target_distribution[:, :, :, i:i+bs-1] |> dev
            (x, _) = model.autoencoder.encode(x)
            latent_target_distribution[:, :, :, i:i+bs-1] = x |> cpu_dev
        end
        trainset_target_distribution = latent_target_distribution

        latent_init_distribution = zeros(L_H, L_W, L_C, len_history, size(trainset_init_distribution)[end])
        for i in 1:batch_size:size(trainset_init_distribution)[end]
            bs = batch_size
            if i + bs - 1 > size(trainset_target_distribution)[end]
                bs = size(trainset_target_distribution)[end] - i + 1
            end
            x = trainset_init_distribution[:, :, :, :, i:i+bs-1] |> dev
            x = reshape(x, H, W, C, len_history*bs)
            (x, _) = model.autoencoder.encode(x)
            x = reshape(x, L_H, L_W, L_C, len_history, bs)
            latent_init_distribution[:, :, :, :, i:i+bs-1] = x |> cpu_dev
        end
        trainset_init_distribution = latent_init_distribution
    end

    init_learning_rate = 1e-4
    new_learning_rate = init_learning_rate

    min_learning_rate = 1e-6

    output_sde = true
    output_ode = false

    early_stop_patience = 10

    num_generator_steps = 75

    num_steps = size(testset, 4)

    num_target = size(trainset_target_distribution)[end]

    best_loss = 1e8

    early_stop_counter = 0

    loss_vec = []
    for epoch in 1:num_epochs
        running_loss = 0.0

        # Shuffle trainset
        train_ids = shuffle(rng, 1:num_target)
        trainset_target_distribution = trainset_target_distribution[:, :, :, train_ids]
        trainset_init_distribution = trainset_init_distribution[:, :, :, :, train_ids]
        trainset_pars_distribution = trainset_pars_distribution[:, train_ids]
    
        for i in 1:batch_size:size(trainset_target_distribution)[end]
            
            if i + batch_size - 1 > size(trainset_target_distribution)[end]
                break
            end

            x_1 = trainset_target_distribution[:, :, :, i:i+batch_size-1] |> dev
            x_0 = trainset_init_distribution[:, :, :, :, i:i+batch_size-1] |> dev
            pars = trainset_pars_distribution[:, i:i+batch_size-1] |> dev
            
            # @infiltrate

            loss, pb_f = Zygote.pullback(
                p -> model.loss(x_0, x_1, pars, p, st, rng, dev)[1], ps
            );

            print(loss)
            error()

            running_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
            
        end

        
        running_loss /= floor(Int, size(trainset_target_distribution)[end] / batch_size)

        loss_vec = vcat(loss_vec, running_loss)

        plot(loss_vec)#, yaxis=:log)
        savefig("training_loss.png")
        
        if epoch % 1 == 0
            print("Loss Value after $epoch iterations: $running_loss \n")
        end
        
        new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ cos.(new_learning_rate ./ num_epochs .* pi));
        Optimisers.adjust!(opt_state, new_learning_rate)


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

                
                # pathwise_MSE = 0
                # mean_MSE = 0
                # x = zeros(size(testset)[1:3]..., num_test_steps, num_test_paths)
                gif_save_path = @sprintf("output/sde_SI_%i", epoch)
                pathwise_MSE, mean_MSE = compare_sde_pred_with_true(
                    model,
                    ps,
                    st_,
                    testset,
                    testset_pars,
                    num_test_paths,
                    normalize_data,
                    mask,
                    num_generator_steps,
                    gif_save_path,
                    rng,
                    dev,
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

                    early_stop_counter = 0

                end

                early_stop_counter += 1

                if early_stop_counter > early_stop_patience
                    println("Early stopping at epoch: ", epoch)
                    println("Best loss: ", best_loss)
                    break
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

                
                # pathwise_MSE = 0
                # mean_MSE = 0
                # x = zeros(size(testset)[1:3]..., num_test_steps, num_test_paths)
                gif_save_path = @sprintf("output/ode_SI_%i", epoch)
                pathwise_MSE, mean_MSE = compare_ode_pred_with_true(
                    model,
                    ps,
                    st_,
                    testset,
                    testset_pars,
                    normalize_data,
                    mask,
                    num_generator_steps,
                    dev,
                    gif_save_path,
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

                    early_stop_counter = 0

                end

                early_stop_counter += 1

                if early_stop_counter > early_stop_patience
                    println("Early stopping at epoch: ", epoch)
                    println("Best loss: ", best_loss)
                    break
                end

                
                CUDA.reclaim()
                GC.gc()
            

            end


        end

    end

    return ps, st

end








