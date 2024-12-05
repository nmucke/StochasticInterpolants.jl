using Lux
using StochasticInterpolants
using Random
using CUDA
using NNlib
using Setfield
using Plots
using Optimisers
using Zygote
using LuxCUDA
using Printf
using Statistics
using ProgressBars
using OrderedCollections

function prepare_latent_data(
    model,
    trainset
)
    (L_H, L_W, L_C) = model.autoencoder.latent_dimensions
    (H, W, C, len_history) = size(trainset.init_distribution)[1:4]

    latent_target_distribution = zeros(L_H, L_W, L_C, size(trainset.target_distribution)[end])
    for i in 1:batch_size:size(trainset.target_distribution)[end]
        bs = batch_size
        if i + bs - 1 > size(trainset.target_distribution)[end]
            bs = size(trainset.target_distribution)[end] - i + 1
        end
        x = trainset.target_distribution[:, :, :, i:i+bs-1] |> dev
        (x, _) = model.autoencoder.encode(x)
        latent_target_distribution[:, :, :, i:i+bs-1] = x |> cpu_dev
    end

    latent_init_distribution = zeros(L_H, L_W, L_C, len_history, size(trainset.init_distribution)[end])
    for i in 1:batch_size:size(trainset.init_distribution)[end]
        bs = batch_size
        if i + bs - 1 > size(trainset.target_distribution)[end]
            bs = size(trainset.target_distribution)[end] - i + 1
        end
        x = trainset_init_distribution[:, :, :, :, i:i+bs-1] |> dev
        x = reshape(x, H, W, C, len_history*bs)
        (x, _) = model.autoencoder.encode(x)
        x = reshape(x, L_H, L_W, L_C, len_history, bs)
        latent_init_distribution[:, :, :, :, i:i+bs-1] = x |> cpu_dev
    end

    trainset = (
        target_distribution=trainset_target_distribution,
        init_distribution=latent_init_distribution,
        pars_distribution=trainset.pars_distribution
    )

    return trainset
end

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
    trainset::NamedTuple,
    testset::NamedTuple,
    checkpoint_manager::NamedTuple=nothing,
    training_args::Dict=nothing,
    normalize_data=nothing,
    mask=nothing,
    rng::AbstractRNG,
    dev=gpu_device()
)
    cpu_dev = LuxCPUDevice();

    # If the model is a LatentFollmerStochasticInterpolant, we need to encode the data
    # before training, to train the SI in the latent space.
    if typeof(model) == LatentFollmerStochasticInterpolant
        trainset = prepare_latent_data(model, trainset)
    end
    

    num_epochs = training_args["num_epochs"]
    batch_size = training_args["batch_size"]
    min_learning_rate = training_args["min_learning_rate"]
    init_learning_rate = training_args["init_learning_rate"]
    new_learning_rate = init_learning_rate
    early_stop_patience = training_args["early_stop_patience"]
    num_test_paths = training_args["num_test_paths"]
    num_test_sde_steps = training_args["num_test_sde_steps"]

    best_loss = 1e8
    early_stop_counter = 0

    loss_vec = []
    loss_dict = OrderedDict()
    test_pathwise_MSE = OrderedDict()
    test_mean_MSE = OrderedDict()
    for epoch in 1:num_epochs
        running_loss = 0.0

        # Shuffle trainset
        train_ids = shuffle(rng, 1:size(trainset.target_distribution)[end])
        trainset = (
            target_distribution=trainset.target_distribution[:, :, :, train_ids],
            init_distribution=trainset.init_distribution[:, :, :, :, train_ids],
            pars_distribution=trainset.pars_distribution[:, train_ids]
        )
    
        for i in 1:batch_size:size(trainset.target_distribution)[end]
            
            if i + batch_size - 1 > size(trainset.target_distribution)[end]
                break
            end

            # Get batch
            x_1 = trainset.target_distribution[:, :, :, i:i+batch_size-1] |> dev
            x_0 = trainset.init_distribution[:, :, :, :, i:i+batch_size-1] |> dev
            pars = trainset.pars_distribution[:, i:i+batch_size-1] |> dev
            
            # Compute loss
            (loss, st), pb_f = Zygote.pullback(
                p -> model.loss(x_0, x_1, pars, p, st, rng, dev), ps
            );
            running_loss += loss

            # Compute gradients
            gs = pb_f((one(loss), nothing))[1];
            
            # Update parameters
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
        end
        
        running_loss /= floor(Int, size(trainset.target_distribution)[end] / batch_size)

        # Save training loss
        loss_vec = vcat(loss_vec, running_loss)
        loss_dict[epoch] = running_loss
        fig = plot(loss_vec, yaxis=:log)
        checkpoint_manager.save_figure(fig, "training_loss.png")
        checkpoint_manager.write_dict_to_txt_file(loss_dict, "training_loss.txt")
        
        print("Loss Value after $epoch iterations: $running_loss \n")
        
        # Adjust learning rate
        new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ 1f0 .* cos.(epoch ./ num_epochs .* pi));
        Optimisers.adjust!(opt_state, new_learning_rate)

        if epoch % training_args["test_frequency"] == 0

            CUDA.reclaim()
            GC.gc()
            
            if !isnothing(normalize_data)
                x_true = normalize_data.inverse_transform(testset.state)
            else
                x_true = testset.state
            end

            # Compute test loss
            pathwise_MSE, mean_MSE = compare_sde_pred_with_true(
                model,
                ps,
                Lux.testmode(st),
                testset.state,
                testset.pars,
                num_test_paths,
                normalize_data,
                mask,
                num_test_sde_steps,
                "$(checkpoint_manager.figures_dir)/$epoch",
                rng,
                dev,
            )

            # Save test loss
            test_pathwise_MSE[epoch] = pathwise_MSE
            test_mean_MSE[epoch] = mean_MSE
            checkpoint_manager.write_dict_to_txt_file(test_pathwise_MSE, "test_pathwise_MSE.txt")
            checkpoint_manager.write_dict_to_txt_file(test_mean_MSE, "test_mean_MSE.txt")

            # Save model if it is the best so far
            if pathwise_MSE < best_loss
                checkpoint_manager.save_model(ps, st, opt_state)
                info_dict = OrderedDict(
                    "Model saved at epoch" => epoch,
                    "test_pathwise_MSE" => pathwise_MSE,
                    "test_mean_MSE" => mean_MSE
                )
                checkpoint_manager.write_dict_to_txt_file(info_dict, "best_model_info.txt")

                best_loss = pathwise_MSE
                early_stop_counter = 0
            end

            # Early stopping
            early_stop_counter += 1
            if early_stop_counter > early_stop_patience
                println("Early stopping at epoch: ", epoch)
                println("Best loss: ", best_loss)
                break
            end

            # Free memory
            CUDA.reclaim()
            GC.gc()
        end
    end
    return ps, st
end




# pathwise_MSE = 0
# mean_MSE = 0
# if output_ode
#     num_test_trajectories = size(testset)[end]
#     num_channels = size(testset, 3)
#     num_test_steps = size(testset, 4)
    
#     if !isnothing(normalize_data)
#         x_true = normalize_data.inverse_transform(testset)
#     else
#         x_true = testset
#     end

    
#     # pathwise_MSE = 0
#     # mean_MSE = 0
#     # x = zeros(size(testset)[1:3]..., num_test_steps, num_test_paths)
#     gif_save_path = @sprintf("output/ode_SI_%i", epoch)
#     pathwise_MSE, mean_MSE = compare_ode_pred_with_true(
#         model,
#         ps,
#         st_,
#         testset,
#         testset_pars,
#         normalize_data,
#         mask,
#         num_generator_steps,
#         dev,
#         gif_save_path,
#     )

#     if !isnothing(model_save_dir) && pathwise_MSE < best_loss
#         save_checkpoint(
#             ps=ps, 
#             st=st,
#             opt_st=opt_state,
#             output_dir=model_save_dir,
#             epoch=epoch
#         )

#         best_loss = pathwise_MSE

#         early_stop_counter = 0

#     end

#     early_stop_counter += 1

#     if early_stop_counter > early_stop_patience
#         println("Early stopping at epoch: ", epoch)
#         println("Best loss: ", best_loss)
#         break
#     end

    
#     CUDA.reclaim()
#     GC.gc()


# end