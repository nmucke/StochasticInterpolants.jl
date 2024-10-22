
using FFTW
using Statistics


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


function compute_divergence(
    sol,
    mask,
)

    num_trajectories = size(sol, 5)
    num_time_steps = size(sol, 4)

    cell_length_x = 1/size(sol, 1)
    cell_length_y = 1/size(sol, 2)

    divergence = zeros(num_time_steps, num_trajectories)
    for i = 1:num_trajectories
        for j in 1:num_time_steps

            u_int = sol[:, :, 1, j, i]
            v_int = sol[:, :, 2, j, i]
        
            # div = (u_int[2:end-1, 2:end-1] - u_int[1:end-2, 2:end-1]) / cell_length_x +
            #         (v_int[2:end-1, 2:end-1] - v_int[2:end-1, 1:end-2]) / cell_length_y

            div = divfunc(u_int, v_int, cell_length_x)

        
            mean_divergence = mean(abs.(div))

            divergence[j, i] = mean_divergence
        end
    end

    return divergence
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


function compute_total_energy(sol)

    num_trajectories = size(sol, 5)
    num_time_steps = size(sol, 4)

    energy = zeros(num_time_steps, num_trajectories)
    for i = 1:num_trajectories
        for j in 1:num_time_steps
            energy[j, i] = sum(sol[:, :, 1, j, i].^2) + sum(sol[:, :, 2, j, i].^2)
            energy[j, i] /= 2
        end
    end

    return energy
end



function compute_energy_spectra(sol)

    num_trajectories = size(sol, 5);
    
    nx = size(sol, 1);
    ny = size(sol, 2);
    
    u = sol[:, :, 1, :];
    v = sol[:, :, 2, :];
    
    u_fft = fft(u, [1, 2]);
    v_fft = fft(v, [1, 2]);
    
    kx = fftfreq(nx, nx);
    ky = fftfreq(ny, ny);
    
    K = (kx.^2)' .+ (ky.^2);
    
    K_bins = logrange(1, maximum(K), 100);
    
    a = 1.6;
    
    u_fft_squared = abs2.(u_fft) ./ (2 * prod(size(u_fft))^2)
    v_fft_squared = abs2.(v_fft) ./ (2 * prod(size(v_fft))^2)
    
    energy = zeros(Float32, length(K_bins), num_trajectories);
    for i = 1:num_trajectories
        for j = 1:length(K_bins)
            
            bin = K_bins[j]

            mask = (K .> bin / a) .& (K .< bin * a)
    
            u_fft_filtered = u_fft_squared[:, :, i] .* mask
            v_fft_filtered = v_fft_squared[:, :, i] .* mask
        
            e = 0.5 * (sum(u_fft_filtered + v_fft_filtered))
    
            energy[j, i] = e
        end
    end
    
    return energy, K_bins
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

        test_init_condition = testset[:, :, :, 1:model.velocity.len_history, i]
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
            mask = mask |> cpu_device()
            x = x .* mask
        end

        mean_pathwise_MSE, mean_mean_MSE = compute_RMSE(
            x_true[:, :, :, :, i], x, mask,
        )

        pathwise_MSE += mean_pathwise_MSE
        mean_MSE += mean_mean_MSE

    end

    # Total energy
    energy_true = compute_total_energy(x_true[:, :, :, :, num_test_trajectories:num_test_trajectories])
    energy_pred = compute_total_energy(x)

    plot(energy_true, color=:blue, label="True", linewidth=3)
    plot!(energy_pred, color=:red, linewidth=3, alpha=0.25)
    plot!(mean(energy_pred, dims=2), color=:red, label="Pred", linewidth=5)
    energy_save_path = gif_save_path * "_energy.pdf"
    savefig(energy_save_path)

    # Divergence
    divergence_true = compute_divergence(x_true[:, :, :, :, num_test_trajectories:num_test_trajectories], mask)
    divergence_pred = compute_divergence(x, mask)

    plot(divergence_pred, color=:red, label="Pred", linewidth=3, alpha=0.25)
    plot!(mean(divergence_pred, dims=2), color=:red, label="Pred", linewidth=5)
    plot!(divergence_true, color=:blue, label="True", linewidth=2)
    divergence_save_path = gif_save_path * "_divergence.pdf"
    savefig(divergence_save_path)

    # Energy spectra
    # energy_spectra_true = compute_energy_spectra(x_true[:, :, :, end, num_test_trajectories:num_test_trajectories])
    # energy_spectra_pred, K_bins = compute_energy_spectra(x[:, :, :, end, 1:1])

    # plot(K_bins, energy_spectra_true, color=:blue, label="True", linewidth=3, xaxis=:log, yaxis=:log)
    # plot!(K_bins, energy_spectra_pred, color=:red, linewidth=3, xaxis=:log, yaxis=:log, alpha=0.25)
    # plot!(K_bins, mean(energy_spectra_pred, dims=2), color=:red, label="Pred", linewidth=5, xaxis=:log, yaxis=:log)
    # energy_spectra_save_path = gif_save_path * "_energy_spectra.pdf"
    # savefig(energy_spectra_save_path)

    true_freq, true_fft = compute_temporal_frequency(x_true[:, :, :, :, num_test_trajectories:num_test_trajectories])
    pred_freq, pred_fft = compute_temporal_frequency(x)
    pred_fft_mean = mean(pred_fft, dims=2)
    pred_fft_min = minimum(pred_fft, dims=2)
    pred_fft_max = maximum(pred_fft, dims=2)

    plot(true_freq, true_fft .* true_fft .* true_fft, color=:blue, label="True", linewidth=3, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft_mean .* pred_fft_mean .* pred_fft_mean, color=:red, label="Pred", linewidth=3, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft_min .* pred_fft_min .* pred_fft_min, linestyle=:dash, color=:red, xaxis=:log2, yaxis=:log10)
    plot!(pred_freq, pred_fft_max .* pred_fft_max .* pred_fft_max, linestyle=:dash, color=:red, xaxis=:log2, yaxis=:log10)

    frequency_save_path = gif_save_path * "_frequency.pdf"
    savefig(frequency_save_path)


    pathwise_MSE /= num_test_trajectories
    mean_MSE /= num_test_trajectories

    println("Mean of pathwise MSE: ", pathwise_MSE)
    println("Mean of mean MSE (SDE): ", mean_MSE)

    x_mean = mean(x, dims=5)[:, :, :, :, 1];
    x_std = std(x, dims=5)[:, :, :, :, 1];

    x_true_plot = sqrt.(x_true[:, :, 1, :, num_test_trajectories].^2 + x_true[:, :, 2, :, num_test_trajectories].^2)
    x_mean_plot = sqrt.(x_mean[:, :, 1, :].^2 + x_mean[:, :, 2, :].^2)
    x_plot = sqrt.(x[:, :, 1, :, :].^2 + x[:, :, 2, :, :].^2)
    x_std_plot = std(x_plot, dims=4)
    # x_true_plot = x_true[:, :, 4, :, num_test_trajectories]
    # x_mean_plot = x_mean[:, :, 4, :]
    # x_plot = x[:, :, 4, :, :]
    # x_std_plot = std(x_plot, dims=4)
    

    # inidividual_snapshot_save_path = gif_save_path * "/individual_snapshots/"
    # for i = 1:num_test_steps
    #     heatmap(
    #         x_plot[:, :, i, 1], legend=false, xticks=false, yticks=false, aspect_ratio=:equal, 
    #         colorbar=true, color=cgrad(:Spectral_11, rev=true), clim=(minimum(x_true_plot), maximum(x_true_plot)),
    #     )
    #     savefig(inidividual_snapshot_save_path * "1_tra_pred_$i.pdf")
    # end
    # for i = 1:num_test_steps
    #     heatmap(
    #         x_plot[:, :, i, 2], legend=false, xticks=false, yticks=false, aspect_ratio=:equal, 
    #         colorbar=true, color=cgrad(:Spectral_11, rev=true), clim=(minimum(x_true_plot), maximum(x_true_plot)),
    #     )
    #     savefig(inidividual_snapshot_save_path * "2_tra_pred_$i.pdf")
    # end
    # for i = 1:num_test_steps
    #     heatmap(
    #         x_plot[:, :, i, 3], legend=false, xticks=false, yticks=false, aspect_ratio=:equal, 
    #         colorbar=true, color=cgrad(:Spectral_11, rev=true), clim=(minimum(x_true_plot), maximum(x_true_plot)),
    #     )
    #     savefig(inidividual_snapshot_save_path * "3_tra_pred_$i.pdf")
    # end
    # for i = 1:num_test_steps
    #     heatmap(
    #         x_std_plot[:, :, i, 1], legend=false, xticks=false, yticks=false, aspect_ratio=:equal, 
    #         colorbar=true, color=cgrad(:Spectral_11, rev=true)
    #     )
    #     savefig(inidividual_snapshot_save_path * "std_$i.pdf")
    # end
    # for i = 1:num_test_steps
    #     heatmap(
    #         x_true_plot[:, :, i], legend=false, xticks=false, yticks=false, aspect_ratio=:equal, 
    #         colorbar=true, color=cgrad(:Spectral_11, rev=true), clim=(minimum(x_true_plot), maximum(x_true_plot)),
    #     )
    #     savefig(inidividual_snapshot_save_path * "true_$i.pdf")
    # end

    preds_to_save = (
        x_true_plot, 
        x_mean_plot, 
        Float16.(abs.(x_mean_plot-x_true_plot)), 
        Float16.(x_std[:, :, end, :]), 
        x_plot[:, :, :, 1], 
        x_plot[:, :, :, 2], 
        x_plot[:, :, :, 3], 
        x_plot[:, :, :, 4]
    );
    create_gif(
        preds_to_save, 
        gif_save_path * ".gif", 
        ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"]
    )

    return pathwise_MSE, mean_MSE

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
    gif_save_path
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
        init_condition=testset[:, :, :, 1:model.velocity.len_history, :],
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
        mask = mask |> cpu_device()
        x = x .* mask
    end

    num_channels = size(x, 3)

    for i = 1:num_test_trajectories
        mean_pathwise_MSE, mean_mean_MSE = compute_RMSE(
            testset[:, :, :, :, i:i], x[:, :, :, :, i:i], mask,
        )

        pathwise_MSE += mean_pathwise_MSE
        mean_MSE += mean_mean_MSE

    end

    x_true = testset[:, :, :, :, num_test_trajectories:num_test_trajectories]

    # Total energy
    energy_true = compute_total_energy(x_true)
    energy_pred = compute_total_energy(x)

    plot(energy_true, color=:blue, label="True", linewidth=3)
    plot!(energy_pred, color=:red, label="Pred", linewidth=3, alpha=0.25)
    plot!(mean(energy_pred, dims=2), color=:red, label="Pred", linewidth=5)
    energy_save_path = gif_save_path * "_energy.pdf"
    savefig(energy_save_path)

    # Divergence
    divergence_true = compute_divergence(x_true, mask)
    divergence_pred = compute_divergence(x, mask)

    plot(divergence_pred, color=:red, label="Pred", linewidth=3, alpha=0.25)
    plot!(mean(divergence_pred, dims=2), color=:red, label="Pred", linewidth=5)
    plot!(divergence_true, color=:blue, label="True", linewidth=2)
    divergence_save_path = gif_save_path * "_divergence.pdf"
    savefig(divergence_save_path)

    # Energy spectra
    energy_spectra_true = compute_energy_spectra(x_true[:, :, :, end, :])
    energy_spectra_pred, K_bins = compute_energy_spectra(x[:, :, :, end, :])

    plot(K_bins, energy_spectra_true, color=:blue, label="True", linewidth=3, xaxis=:log, yaxis=:log)
    plot!(K_bins, energy_spectra_pred, color=:red, label="Pred", linewidth=3, xaxis=:log, yaxis=:log, alpha=0.25)
    plot!(K_bins, mean(energy_spectra_pred, dims=2), color=:red, label="Pred", linewidth=5, xaxis=:log, yaxis=:log)
    energy_spectra_save_path = gif_save_path * "_energy_spectra.pdf"
    savefig(energy_spectra_save_path)

    
    # true_freq, true_fft = compute_temporal_frequency(x_true[:, :, :, :, num_test_trajectories:num_test_trajectories])
    # pred_freq, pred_fft = compute_temporal_frequency(x)
    # pred_fft = pred_fft[:, num_test_trajectories]

    # plot(true_freq, true_fft .* true_fft .* true_fft, color=:blue, label="True", linewidth=3, xaxis=:log2, yaxis=:log10)
    # plot!(pred_freq, pred_fft .* pred_fft .* pred_fft, color=:red, label="Pred", linewidth=3, xaxis=:log2, yaxis=:log10)
    
    # frequency_save_path = @sprintf("output/ode_frequency_%i.pdf", epoch)
    # savefig(frequency_save_path)

    
    x_pred_plot = sqrt.(x[:, :, 1, :, 1].^2 + x[:, :, 2, :, 1].^2)
    x_true_plot = sqrt.(x_true[:, :, 1, :, 1].^2 + x_true[:, :, 2, :, 1].^2)
        
    preds_to_save = (x_true_plot, x_pred_plot, x_pred_plot-x_true_plot)
    create_gif(
        preds_to_save, 
        gif_save_path * ".gif",
        ["True", "Pred", "Error"]
    )

    println("MSE (ODE): ", pathwise_MSE)
    
    CUDA.reclaim()
    GC.gc()
end