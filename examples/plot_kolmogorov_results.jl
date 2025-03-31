ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "20GiB"
ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"

using JLD2
using StochasticInterpolants
using YAML
using NPZ
using FileIO
using Plots
using Random
using Statistics
using KernelDensity
using Distances
using Divergences
using LaTeXStrings
using GLMakie
using CairoMakie
# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

tickfontsize = 10
tickfontsize = 10
legendfontsize = 10
guidefontsize = 10



# Choose between "incompressible_flow", "kolmogorov"
test_case = "kolmogorov";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "default";

_, _, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);

num_trajectories = size(testset, 5);
num_steps = size(testset, 4);
H, W, C = size(testset, 1), size(testset, 2), size(testset, 3);

num_generator_steps_list = [10, 25, 50];
refiner_steps_list = [2, 4, 8];
results_dir = "results/";
cases_to_plot = [
    "SI_not_optimized_",
    "SI_not_optimized_with_projection_",
    "SI_optimized_no_projection_",
    "SI_optimized_with_projection_",
    "acdm_",
    "refiner_",
]
labels = [
    "Filtered DNS",
    L"SI", #"Not Optimized",
    L"SI_{div}", #"Not Optimized with Projection",
    L"SI_{opt}", #"Optimized",
    L"SI_{opt, div}", #"Optimized with Projection",
    L"ACDM",
    L"Refiner",
]




times_to_plot = [10, 50, 100, 200, 400, 750]
dt = 0.01

# preds = Dict()
preds_energy_spectra = Dict()
preds_total_energy = Dict()
pred_correlation = Dict()
pred_rate_of_change_magnitude = Dict()
pred_mse = Dict()
pred_KL_divergence = Dict()
preds_to_plot = Dict(
    time_step => Dict() for time_step in times_to_plot
)

true_energy_spectra = [compute_energy_spectra(testset[:, :, :, ti, :])[1] for ti in 1:num_steps];
true_energy_spectra = cat(true_energy_spectra..., dims=3);
true_energy_spectra = permutedims(true_energy_spectra, (1, 3, 2));
true_total_energy = compute_total_energy(testset);
true_rate_of_change_magnitude = mean(abs.(testset[:, :, :, 2:end, :] .- testset[:, :, :, 1:end-1, :])./dt, dims=(1,2,3))[1, 1, 1, :, :];

save("$(results_dir)npz/true_data_total_energy.npz", Dict("arr_0" => true_total_energy))


pred_vel_magnitude = sqrt.(testset[:, :, 1, :, 1].^2 .+ testset[:, :, 2, :, 1].^2);
plot_list = [pred_vel_magnitude];
create_gif(plot_list, "animations/true_data", [""])
# save("$(results_dir)npz/true_data.npz", Dict("arr_0" => testset))

for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = refiner_steps_list
    else
        _num_generator_steps_list = num_generator_steps_list
    end
    for num_generator_steps in _num_generator_steps_list
        pred = load("$results_dir$case$num_generator_steps.jld2", "data"); # H, W, C, num_steps, nums_paths, num_test

        # Save prediction data as npz
        npzwrite("$(results_dir)npz/$case$num_generator_steps.npz", Dict("arr_0" => pred))

        for time_step in times_to_plot
            preds_to_plot[time_step]["$case$num_generator_steps"] = sqrt.(pred[:, :, 1, time_step, :, 1].^2 .+ pred[:, :, 2, time_step, :, 1].^2);
        end

        preds_energy_spectra["$case$num_generator_steps"] = get_energy_spectra(pred); # num_bins, num_steps, num_paths, num_trajectories
        preds_total_energy["$case$num_generator_steps"] = get_total_energy(pred); # num_steps, num_paths, num_trajectories
        pred_mse["$case$num_generator_steps"] = get_mse(pred[:, :, :, 1:50, :, :], testset[:, :, :, 1:50, :, :]);
        # pred_KL_divergence["$case$num_generator_steps"] = get_KL_divergence(
        #     preds_total_energy["$case$num_generator_steps"], 
        #     true_total_energy
        # );
        pred_correlation["$case$num_generator_steps"] = get_pearson_correlation(pred, testset);
        pred_rate_of_change_magnitude["$case$num_generator_steps"] = mean(abs.(pred[:, :, :, 2:end, :, :] .- pred[:, :, :, 1:end-1, :, :])./dt, dims=(1,2,3))[1, 1, 1, :, :, :];


        save("$(results_dir)npz/$(case)$(num_generator_steps)_total_energy.npz", Dict("arr_0" => preds_total_energy["$case$num_generator_steps"]))

        println("Loaded $case$num_generator_steps")
    end
end


figure_folder = "figures_for_paper";
for (i, case) in enumerate(cases_to_plot)
    println("Plotting $case")

    if case == "refiner_"
        _num_generator_steps_list = refiner_steps_list
    else
        _num_generator_steps_list = num_generator_steps_list;
    end

    label = labels[i+1]
    plot_labels = (label * " $num_generator_steps" for num_generator_steps in _num_generator_steps_list)
    plot_labels = ("Filtered DNS", plot_labels...)

    for time_step = times_to_plot
        plot_list = (preds_energy_spectra["$case$num_generator_steps"][:, time_step, :, :] for num_generator_steps in _num_generator_steps_list);
        y_lim_max = 5e2
        plot_list = (true_energy_spectra[:, time_step, :], plot_list..., );
        plot_energy_spectra(;
            spectra=plot_list, 
            labels=plot_labels, 
            with_ci=true,
            ylims=(1e-8, y_lim_max*1.1)
        )
        savefig("$figure_folder/energy_spectra_$(case)time_step_$(time_step).pdf")
    end
    
    plot_list = (preds_total_energy["$case$num_generator_steps"] for num_generator_steps in _num_generator_steps_list);
    plot_list = (true_total_energy, plot_list..., );
    plot_total_energy(;
        total_energy=plot_list, 
        labels=plot_labels, 
        with_ci=true,
        ylabel="Total Energy",
        plot_mean=true
    )
    savefig("$figure_folder/total_energy_$(case).pdf")
    
    plot_list = (vcat(energy...) for energy in plot_list);
    plot_energy_density(;
        energies=plot_list, 
        labels=plot_labels, 
    )
    savefig("$figure_folder/energy_density_$(case).pdf")
    
    
    plot_list = (pred_rate_of_change_magnitude["$case$num_generator_steps"] for num_generator_steps in _num_generator_steps_list);
    plot_list = (true_rate_of_change_magnitude, plot_list..., );
    plot_total_energy(;
        total_energy=plot_list, 
        labels=plot_labels, 
        with_ci=true,
        ylims=(0, 15),
        ylabel="Rate of change",
        plot_mean=true
    )
    savefig("$figure_folder/rate_of_change_magnitude_$(case).pdf")


    plot_list = (pred_correlation["$case$num_generator_steps"] for num_generator_steps in _num_generator_steps_list);
    plot_total_energy(;
        total_energy=plot_list, 
        labels=plot_labels[2:end], 
        with_ci=true,
        ylims=(0, 1.2),
        ylabel="Pearson Correlation",
        plot_mean=true
    )
    savefig("$figure_folder/pearson_correlation_$(case).pdf")
end


figure_folder = "figures_for_paper_comparison";


cases_to_compare = ["acdm_50", "refiner_8", "SI_not_optimized_50", "SI_optimized_with_projection_50"]
labels_to_compare = [
    "Filtered DNS",
    "ACDM, 50",
    "Refiner, 8",
    L"SI, 50", #"Not Optimized",
    L"SI_{opt, div}, 50", #"Optimized with Projection",
]



for time_step = times_to_plot
    plot_list = (preds_energy_spectra[case][:, time_step, :, :] for case in cases_to_compare);
    plot_list = (true_energy_spectra[:, time_step, :], plot_list..., );
    y_lim_max = 5e2
    plot_energy_spectra(;
        spectra=plot_list, 
        labels=labels_to_compare, 
        with_ci=true,
        ylims=(1e-8, y_lim_max*1.1)
    )
    savefig("$figure_folder/energy_spectra_comparison_time_step_$(time_step).pdf")
end

plot_list = (preds_total_energy[case] for case in cases_to_compare);
plot_list = (true_total_energy, plot_list..., );
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels_to_compare, 
    with_ci=true,
    ylabel="Total Energy",
    plot_mean=true
)
savefig("$figure_folder/total_energy_comparison.pdf")

plot_list = (vcat(energy...) for energy in plot_list);
plot_energy_density(;
    energies=plot_list, 
    labels=labels_to_compare, 
)
savefig("$figure_folder/energy_density_comparison.pdf")


plot_list = (pred_rate_of_change_magnitude[case] for case in cases_to_compare);
plot_list = (true_rate_of_change_magnitude, plot_list..., );
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels_to_compare, 
    with_ci=true,
    ylims=(0, 15),
    ylabel="Rate of change",
    plot_mean=true
)
savefig("$figure_folder/rate_of_change_magnitude_comparison.pdf")


plot_list = (pred_correlation[case] for case in cases_to_compare);
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels_to_compare[2:end], 
    with_ci=true,
    ylims=(0, 1.2),
    ylabel="Pearson Correlation",
    plot_mean=true
)
savefig("$figure_folder/pearson_correlation_comparison.pdf")


for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = refiner_steps_list
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        
        pred_cor = pred_correlation["$case$num_generator_steps"]
        time_to_below_0_8 = pred_cor .> 0.8

        # Count number of elements until first false (correlation drops below 0.8)
        time_to_decorrelate = zeros(Int, size(time_to_below_0_8)[2:end]...)
        for i in CartesianIndices(time_to_decorrelate)
            time_series = time_to_below_0_8[:, i]
            # Find first index where correlation drops below 0.8
            first_false = findfirst(x -> !x, time_series)
            if isnothing(first_false)
                time_to_decorrelate[i] = length(time_series)
            else
                time_to_decorrelate[i] = first_false - 1
            end
        end

        time_to_decorrelate = dt * time_to_decorrelate

        println("Time to decorrelate for $case$num_generator_steps: ", round(mean(time_to_decorrelate), digits=3), " \$\\pm\$ ", round(std(time_to_decorrelate), digits=3))

    end
end


# for case in cases_to_plot

#     if case == "refiner_"
#         _num_generator_steps_list = refiner_steps_list
#     else
#         _num_generator_steps_list = num_generator_steps_list;
#     end

#     for num_generator_steps in _num_generator_steps_list
#         pred = load("$results_dir$case$num_generator_steps.jld2", "data"); # H, W, C, num_steps, nums_paths, num_test
#         pred_vel_magnitude = sqrt.(pred[:, :, 1, :, :, 1].^2 .+ pred[:, :, 2, :, :, 1].^2);
#         plot_list = [pred_vel_magnitude[:, :, :, i] for i in 1:5];
#         create_gif(plot_list, "animations/$case$num_generator_steps", ["", "", "", "", ""])
        
#         println("Case $case$num_generator_steps done")
#     end
# end


for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = refiner_steps_list
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        println("MSE for $case$num_generator_steps: ", round(mean(pred_mse["$case$num_generator_steps"]), digits=3), " \$\\pm\$ ", round(std(pred_mse["$case$num_generator_steps"]), digits=3))
    end
end

for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = refiner_steps_list
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        println("MSE for $case$num_generator_steps: ", round(std(pred_mse["$case$num_generator_steps"]), digits=3))
    end
end


velocity_magnitude_true = sqrt.(testset[:, :, 1, :, 1].^2 .+ testset[:, :, 2, :, 1].^2);
n_plots = length(times_to_plot)
fig_width = 400 * n_plots # 200 pixels per plot
fig_height = 400 # Keep height constant
fig = CairoMakie.Figure(; size = (fig_width, fig_height))
hm = nothing
for (i, time_step) in enumerate(times_to_plot)
    if i == 1
        y_label = "Filtered DNS"
    else
        y_label = ""
    end
    ax = Axis(
        fig[1, i];
        # title = "Time step $time_step",
        aspect = DataAspect(),
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        ylabel = y_label,
        ylabelsize = 40,
        # titlesize = 40,
    )
    
    hm = CairoMakie.heatmap!(
        ax, 
        velocity_magnitude_true[:, :, time_step]; 
        colormap = Reverse(:Spectral_11),
        # clim = (minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)),
        colorrange = (minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)),
    )
end
Colorbar(fig[1, end+1], hm, ticklabelsize=25, width=30)
CairoMakie.save("$figure_folder/velocity_magnitude_true.pdf", fig)


for (j, case) in enumerate(cases_to_plot)
    if case == "refiner_"
        _num_generator_steps_list = [8]
    else
        _num_generator_steps_list = num_generator_steps_list
    end
    for num_generator_steps in _num_generator_steps_list
        n_plots = length(times_to_plot)
        fig_width = 400 * n_plots 
        fig_height = 400 
        fig = CairoMakie.Figure(; size = (fig_width, fig_height))
        for (i, time_step) in enumerate(times_to_plot)
            if i == 1
                y_label = labels[j+1]
                # y_label = y_label * L" $num_generator_steps"
            else
                y_label = ""
            end
            ax = Axis(
                fig[1, i];
                aspect = DataAspect(),
                xticksvisible = false,
                xticklabelsvisible = false,
                yticksvisible = false,
                yticklabelsvisible = false,
                ylabel = y_label,
                ylabelsize = 40,
            )
            velocity_magnitude = preds_to_plot[time_step]["$case$num_generator_steps"][:, :, 1]
            GLMakie.heatmap!(ax, velocity_magnitude; colormap = Reverse(:Spectral_11), colorrange = (minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)))

        end
        Colorbar(fig[1, end+1], hm,  ticklabelsize=25, width=30)
        CairoMakie.save("$figure_folder/velocity_magnitude_$(case)$(num_generator_steps).pdf", fig)
    end
end



for (j, case) in enumerate(cases_to_plot)
    if case == "refiner_"
        _num_generator_steps_list = [8]
    else
        _num_generator_steps_list = [50]
    end
    for num_generator_steps in _num_generator_steps_list
        for (t_i, time_step) in enumerate([times_to_plot[3], times_to_plot[end]])
            n_plots = length(5)
            fig_width = 400 * 5 
            fig_height = 400
            fig = CairoMakie.Figure(; size = (fig_width, fig_height), )
            for (i, path_i) in enumerate(1:5)
                if i == 1
                    y_label = "Time step $time_step"
                    # y_label = y_label * L" $num_generator_steps"
                else
                    y_label = ""
                end
                ax = Axis(
                    fig[1, i];
                    aspect = DataAspect(),
                    xticksvisible = false,
                    xticklabelsvisible = false,
                    yticksvisible = false,
                    yticklabelsvisible = false,
                    ylabel = y_label,
                    ylabelsize = 40,
                )
                velocity_magnitude = preds_to_plot[time_step]["$case$num_generator_steps"][:, :, path_i]
                hm = GLMakie.heatmap!(
                    ax, 
                    velocity_magnitude; 
                    colormap = Reverse(:Spectral_11), 
                    colorrange = (minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)),
                )
            end
            Colorbar(fig[1, end+1], hm, ticklabelsize=25, width=30)
            CairoMakie.save("$figure_folder/velocity_magnitude_$(case)$(num_generator_steps)_time_step_$(time_step)_paths.pdf", fig)
        end
    end
end




for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = [8]
    else
        _num_generator_steps_list = [50]
    end
    for num_generator_steps in _num_generator_steps_list
        n_plots = length(times_to_plot)
        fig_width = 400 * n_plots 
        fig_height = 400 
        fig = Figure(; size = (fig_width, fig_height))
        for (i, path_i) in enumerate(1:5)
            if i == 1
                y_label = labels[i+1]
            else
                y_label = ""
            end
            ax = Axis(
                fig[1, i];
                # title = "Path $path_i",
                aspect = DataAspect(),
                xticksvisible = false,
                xticklabelsvisible = false,
                yticksvisible = false,
                yticklabelsvisible = false,
                ylabel = y_label,
            )
            velocity_magnitude = preds_to_plot[times_to_plot[end]]["$case$num_generator_steps"][:, :, 1]
            GLMakie.heatmap!(ax, velocity_magnitude; colormap = Reverse(:Spectral_11))
            savefig("$figure_folder/velocity_magnitude_$(case)$(num_generator_steps)_path_$(path_i).pdf")
        end
    end
end



for time_step in times_to_plot
    for case in cases_to_plot
        if case == "refiner_"
            _num_generator_steps_list = [8]
        else
            _num_generator_steps_list = num_generator_steps_list
        end
        for num_generator_steps in _num_generator_steps_list
            for path_i in range(1, 5)
                velocity_magnitude = preds_to_plot[time_step]["$case$num_generator_steps"][:, :, path_i]
                Plots.heatmap(
                    velocity_magnitude, 
                    aspect_ratio=:equal, 
                    legend=false, 
                    xticks=false, 
                    yticks=false, 
                    clim=(minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)), 
                    colormap=cgrad(:Spectral_11, rev=true),
                    size=(400, 400),
                )
                savefig("$figure_folder/velocity_magnitude_$(case)$(num_generator_steps)_time_step_$(time_step)_path_$(path_i).pdf")
            end
        end
    end
end







