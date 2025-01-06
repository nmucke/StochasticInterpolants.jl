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


function get_value_or_default(dict, key, default)
    if haskey(dict, key)
        return dict[key]
    else
        return default
    end
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

    num_epochs = get_value_or_default(training_args, "num_epochs", 1000)
    batch_size = get_value_or_default(training_args, "batch_size", 8)
    min_learning_rate = get_value_or_default(training_args, "min_learning_rate", 1e-6)
    new_learning_rate = min_learning_rate
    max_learning_rate = get_value_or_default(training_args, "max_learning_rate", 1e-4)
    early_stop_patience = get_value_or_default(training_args, "early_stop_patience", 10)
    num_test_paths = get_value_or_default(training_args, "num_test_paths", 5)
    num_test_sde_steps =  get_value_or_default(training_args, "num_test_sde_steps", 50)
    num_warmup_steps =  get_value_or_default(training_args, "num_warmup_steps", 10)

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
        if epoch < num_warmup_steps
            new_learning_rate = epoch * max_learning_rate / num_warmup_steps
        else
            new_learning_rate = min_learning_rate .+ 0.5f0 .* (max_learning_rate - min_learning_rate) .* (1 .+ 1f0 .* cos.((epoch-num_warmup_steps) ./ num_epochs .* pi));
        end
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



"""
    train_stochastic_interpolant_for_closure(
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
function train_stochastic_interpolant_for_closure(;
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
    test_RRMSE = OrderedDict()
    test_MAPE = OrderedDict()
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
            x_condition = trainset.init_distribution[:, :, :, :, i:i+batch_size-1] |> dev
            pars = trainset.pars_distribution[:, i:i+batch_size-1] |> dev
            
            # Compute loss
            (loss, st), pb_f = Zygote.pullback(
                p -> model.loss(x_condition, x_1, pars, p, st, rng, dev), ps
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

            test_pred = forecasting_sde_sampler(
                testset.init_distribution |> dev,
                testset.pars |> dev,
                model,
                ps,
                Lux.testmode(st),
                num_test_sde_steps,
                rng,
                dev
            ) |> cpu_dev


            # Compute test loss
            mean_MSE = mean((test_pred .- testset.target_distribution).^2)
            RRMSE = sqrt(mean_MSE) / sqrt(mean(testset.target_distribution.^2))
            MAPE = mean(abs.(test_pred .- testset.target_distribution ./ testset.target_distribution))

            print("Mean MSE: $mean_MSE \n")
            print("RRMSE: $RRMSE \n")
            print("MAPE: $MAPE \n")

            # Save test loss
            test_mean_MSE[epoch] = mean_MSE
            test_RRMSE[epoch] = RRMSE
            test_MAPE[epoch] = MAPE
            checkpoint_manager.write_dict_to_txt_file(test_mean_MSE, "test_mean_MSE.txt")
            checkpoint_manager.write_dict_to_txt_file(test_RRMSE, "test_RRMSE.txt")
            checkpoint_manager.write_dict_to_txt_file(test_MAPE, "test_MAPE.txt")

            # Save model if it is the best so far
            if mean_MSE < best_loss
                checkpoint_manager.save_model(ps, st, opt_state)
                info_dict = OrderedDict(
                    "Model saved at epoch" => epoch,
                    "test_mean_MSE" => mean_MSE,
                    "test_RRMSE" => RRMSE,
                    "test_MAPE" => MAPE
                )
                checkpoint_manager.write_dict_to_txt_file(info_dict, "best_model_info.txt")

                best_loss = mean_MSE
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