
"""
    compute_SDE_trajectories_RMSE(
        testset_state,
        testset_pars,
        model,
        ps,
        st,
        num_generator_steps,
        num_test_paths,
        normalize_data,
        mask,
        rng,
        dev,
        gif_save_path=nothing,
    )

Compute multiple steps of the SDE model and compare the results to the true data.
"""

function compute_SDE_trajectories_RMSE(
    testset_state,
    testset_pars,
    model,
    ps,
    st,
    num_generator_steps,
    num_test_paths,
    normalize_data,
    mask,
    rng,
    dev,
    gif_save_path=nothing,
)
    
    num_test_paths = num_test_paths |> dev;

    num_test_trajectories = size(testset_state)[end];
    num_channels = size(testset_state, 3);
    num_test_steps = size(testset_state, 4);

    st_ = Lux.testmode(st);

    if !isnothing(normalize_data)
        x_true = normalize_data.inverse_transform(testset_state)
    else
        x_true = testset_state
    end;

    if !isnothing(mask)
        x_true = x_true .* mask
        num_non_obstacle_grid_points = sum(mask)
    else
        num_non_obstacle_grid_points = size(x_true)[1] * size(x_true)[2]
    end;

    pathwise_MSE = []
    mean_MSE = []
    x = zeros_like(testset_state);
    for i = 1:num_test_trajectories

        test_init_condition = testset_state[:, :, :, 1:1, i]
        test_pars = testset_pars[:, 1:1, i]

        x = compute_multiple_SDE_steps(
            init_condition=test_init_condition,
            parameters=test_pars,
            num_physical_steps=num_test_steps,
            num_generator_steps=num_generator_steps,
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

        error_i = 0
        for j = 1:num_test_paths
            error_i += sum((x[:, :, :, :, j] - x_true[:, :, :, :, i]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
        end
        error_i = error_i / num_test_paths

        push!(pathwise_MSE, error_i)

        x_mean = mean(x, dims=5)[:, :, :, :, 1]
        x_std = std(x, dims=5)[:, :, :, :, 1]

        MSE = sum((x_mean - x_true[:, :, :, :, i]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
        push!(mean_MSE, MSE)
    end;

    mean_pathwise_MSE = mean(pathwise_MSE)
    mean_mean_MSE = mean(mean_MSE)
    
    # println("Mean of pathwise MSE: ", mean_pathwise_MSE)
    # println("Std of pathwise MSE: ", std(pathwise_MSE))

    # println("Mean of mean MSE (SDE): ", mean_mean_MSE)
    # println("Std of mean MSE (SDE): ", std(mean_MSE))

    x_mean = mean(x, dims=5)[:, :, :, :, 1];
    x_std = std(x, dims=5)[:, :, :, :, 1];

    x_true = x_true[:, :, :, :, num_test_trajectories];

    if !isnothing(gif_save_path)
        preds_to_save = (
            x_true[:, :, 4, :], 
            x_mean[:, :, 4, :], 
            Float16.(x_mean[:, :, 4, :]-x_true[:, :, 4, :]), 
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
    end;

    CUDA.reclaim()
    GC.gc()

    return mean_pathwise_MSE, mean_mean_MSE

end


function compute_ODE_trajectories_RMSE(
    testset_state,
    testset_pars,
    model,
    ps,
    st,
    normalize_data,
    mask,
    dev,
    gif_save_path=nothing,
)

    num_test_trajectories = size(testset_state)[end];
    num_channels = size(testset_state, 3);
    num_test_steps = size(testset_state, 4);

    st_ = Lux.testmode(st);

    if !isnothing(normalize_data)
        x_true = normalize_data.inverse_transform(testset_state)
    else
        x_true = testset_state
    end;

    if !isnothing(mask)
        x_true = x_true .* mask
        num_non_obstacle_grid_points = sum(mask)
    else
        num_non_obstacle_grid_points = size(x_true)[1] * size(x_true)[2]
    end;

    pathwise_MSE = []
    x = zeros_like(testset_state);
    for i = 1:num_test_trajectories
        test_init_condition = testset_state[:, :, :, 1:1, i]
        test_pars = testset_pars[:, 1:1, i]

        x = compute_multiple_ODE_steps(
            init_condition=test_init_condition,
            parameters=test_pars,
            num_physical_steps=num_test_steps,
            num_generator_steps=25,
            model=model,
            ps=ps,
            st=st_,
            dev=dev,
            mask=mask,
        )

        num_channels = size(x, 3)
        
        if !isnothing(normalize_data)
            x = normalize_data.inverse_transform(x)
            x_true = normalize_data.inverse_transform(testset)
        end
        
        if !isnothing(mask)
            x = x .* mask
            x_true = x_true .* mask
        
            num_non_obstacle_grid_points = sum(mask)
        end
        
        MSE = sum((x[:, :, :, :, 1] - x_true[:, :, :, :, 1]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels

        push!(pathwise_MSE, MSE)

    end;

    mean_pathwise_MSE = mean(pathwise_MSE)
            
    x = x[:, :, 4, :, 1]
    x_true = x_true[:, :, 4, :, 1]
    
    if !isnothing(gif_save_path)
        preds_to_save = (x_true, x, x-x_true)
        create_gif(preds_to_save, gif_save_path, ["True", "Pred", "Error"])
    end;
    
    CUDA.reclaim()
    GC.gc()

    return mean_pathwise_MSE
end
