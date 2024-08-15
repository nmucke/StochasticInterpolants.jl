
"""
    compute_SDE_trajectories_RMSE(
        true_sol,
        pred_sol,
        mask,
        gif_save_path=nothing,
    )

Compute multiple steps of the SDE model and compare the results to the true data.
"""

function compute_RMSE(
    true_sol,
    pred_sol,
    mask,
    gif_save_path=nothing,
)

    if !isnothing(mask)
        x_true = x_true .* mask
        num_non_obstacle_grid_points = sum(mask)
    else
        num_non_obstacle_grid_points = size(x_true)[1] * size(x_true)[2]
    end

    pathwise_MSE = []
    for i = 1:size(pred_sol, 5)
        error_i = sum((pred_sol[:, :, :, :, i] - true_sol).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
        push!(pathwise_MSE, error_i)
    end

    x_mean = mean(x, dims=5)[:, :, :, :, 1]

    mean_MSE = sum((x_mean - true_sol).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels

    mean_pathwise_MSE = mean(pathwise_MSE)

    return mean_pathwise_MSE, mean_MSE

end



function compute_spatial_frequency()

end


function compute_temporal_frequency()

end

