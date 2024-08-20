
using FFTW


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
)

    num_test_steps = size(true_sol)[4]
    num_channels = size(true_sol)[3]

    if !isnothing(mask)
        true_sol = true_sol .* mask
        num_non_obstacle_grid_points = sum(mask)
    else
        num_non_obstacle_grid_points = size(true_sol)[1] * size(true_sol)[2]
    end

    pathwise_MSE = []
    for i = 1:size(pred_sol, 5)
        error_i = sum((pred_sol[:, :, :, :, i] - true_sol).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
        push!(pathwise_MSE, error_i)
    end

    pred_mean = mean(pred_sol, dims=5)[:, :, :, :, 1]

    mean_MSE = sum((pred_mean - true_sol).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels

    mean_pathwise_MSE = mean(pathwise_MSE)

    return mean_pathwise_MSE, mean_MSE

end



function compute_spatial_frequency()

end


function compute_temporal_frequency(
    sol; # (H, W, C, T, B)
    grid_spacing = 0.002, # delta t between frames from simulation
    field = 2, # velocity_x (1), velocity_y (2), density (3), or pressure (4)
    frac_h = 0.5, # vertically centered
    frac_w = 0.25, # closely behing the cylinder
)

    num_trajectories = size(sol, 5)
    num_time_steps = size(sol, 4)


    pos_h = Int(frac_h * size(sol, 1))
    pos_w = Int(frac_w * size(sol, 2))

    # true_sol = true_sol[pos_h, pos_w, field, :, :] # (T, B)
    sol = sol[pos_h, pos_w, field, :, :] # (T, B)

    sol_fft = zeros(num_time_steps, num_trajectories)

    for i = 1:num_trajectories
        out = fft(sol[:, i])
        out = real(out .* conj(out))

        sol_fft[:, i] = out
    end

    n = size(sol_fft, 1)

    freq = fftfreq(n, grid_spacing)[2:Int(n/2)]
    sol_fft = sol_fft[2:Int(n/2), :] # only use positive fourier frequencies

    return freq, sol_fft
end



function compare_sde_pred_with_true(
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

    num_test_trajectories = size(testset)[end]
    num_channels = size(testset, 3)
    num_test_steps = size(testset, 4)

    if !isnothing(normalize_data)
        x_true = normalize_data.inverse_transform(testset)
    else
        x_true = testset
    end

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

    

    plot(true_freq, true_fft .* true_fft .* true_fft, color=:blue, label="True", linewidth=3, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft_mean .* pred_fft_mean .* pred_fft_mean, color=:red, label="Pred", linewidth=3, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft_min .* pred_fft_min .* pred_fft_min, linestyle=:dash, color=:red, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft_max .* pred_fft_max .* pred_fft_max, linestyle=:dash, color=:red, xaxis=:log2, yaxis=:log10)

    frequency_save_path = gif_save_path * "_frequency.png"
    savefig(frequency_save_path)


    pathwise_MSE /= num_test_trajectories
    mean_MSE /= num_test_trajectories

    println("Mean of pathwise MSE: ", pathwise_MSE)
    println("Mean of mean MSE (SDE): ", mean_MSE)

    x_mean = mean(x, dims=5)[:, :, :, :, 1];
    x_std = std(x, dims=5)[:, :, :, :, 1];

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
        gif_save_path * ".gif", 
        ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"]
    )

end



function compare_ode_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    normalize_data,
    mask,
    num_generator_steps,
    dev,
    epoch,
    pred_fft_mean,
)

    pathwise_MSE = 0
    mean_MSE = 0

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
        num_generator_steps=num_generator_steps,
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

    plot(true_freq, true_fft .* true_fft .* true_fft, color=:blue, label="True", linewidth=3, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft .* pred_fft .* pred_fft_mean, color=:red, label="Pred", linewidth=3, xaxis=:log2, yaxis=:log10)
    
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