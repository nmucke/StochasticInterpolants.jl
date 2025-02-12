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
# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);


# H, W, C, num_steps, nums_paths, num_test
function get_pearson_correlation(pred, _true)
    H, W, C, num_steps, nums_paths, num_trajectories = size(pred)
    
    pred_mean = mean(pred, dims=(1, 2, 3))
    true_mean = mean(_true, dims=(1, 2, 3))
    
    pred_std = std(pred, dims=(1, 2, 3), corrected=false)
    true_std = std(_true, dims=(1, 2, 3), corrected=false)
    
    corr = mean((input .- input_mean) .* (target .- target_mean), dims=1) ./ 
           max.(pred_std .* true_std, floatmin(Float32))
    
    corr = dropdims(corr, dims=1) # Shape (T, B)
    
    corr = mean(corr, dims=2) # Reduce over batch
    corr = dropdims(corr, dims=2) # Shape (T)
    
    return corr
end


function plot_energy_density(; energies, labels)

    color_list = palette(:tab10);

    x = LinRange(0, 100, 2048)
    p = plot(x_lims=(0, 100), xlabel="Energy", ylabel="P(energy)", normed=true)
    counter = 1
    for energy in energies
        k = kde(energy)
        ik = InterpKDE(k)
        z = pdf(ik, x)
        plot!(p, x, z, label=labels[counter], linewidth=3, color=color_list[counter])
        plot!(p, x, z, fillrange=0, alpha=0.2, color=color_list[counter], linewidth=0, label="")
        counter += 1
    end
    return p
end


function plot_energy_spectra(; spectra, labels, with_ci=false)

    color_list = palette(:tab10);

    p = plot(ylims = (1e-8, 500000), xaxis=:log, yaxis=:log, xlabel="Frequency", ylabel="Energy");

    counter = 1;
    for spectrum in spectra
        if length(size(spectrum)) == 3
            mean_spectrum = mean(spectrum, dims=(2, 3))[:,1,1];
            std_spectrum = std(spectrum, dims=(2, 3))[:,1,1];
            # Compute 5th and 95th percentiles along dimensions 2 and 3
            lower_bound = mapslices(x -> quantile(vec(x), 0.05), spectrum, dims=[2,3])[:,1,1]
            upper_bound = mapslices(x -> quantile(vec(x), 0.95), spectrum, dims=[2,3])[:,1,1]
            # lower_bound = minimum(spectrum, dims=(2, 3))
            # upper_bound = maximum(spectrum, dims=(2, 3))
        else
            mean_spectrum = mean(spectrum, dims=2)[:,1];
            std_spectrum = std(spectrum, dims=2)[:,1];
            lower_bound = mapslices(x -> quantile(vec(x), 0.05), spectrum, dims=2)[:,1]
            upper_bound = mapslices(x -> quantile(vec(x), 0.95), spectrum, dims=2)[:,1]
            # lower_bound = minimum(spectrum, dims=2)
            # upper_bound = maximum(spectrum, dims=2)
        end

        mean_spectrum = filter(!iszero, mean_spectrum)
        std_spectrum = filter(!iszero, std_spectrum)

        lower_bound = filter(!iszero, lower_bound)
        upper_bound = filter(!iszero, upper_bound)

        plot!(p, mean_spectrum, label=labels[counter], linewidth=3, color=color_list[counter])
        # plot!(p, lower_bound, alpha=0.2, color=color_list[counter], linewidth=5, label="")
        if with_ci
            plot!(p, lower_bound, fillrange=upper_bound, alpha=0.2, color=color_list[counter], linewidth=0, label="")
        end

        counter += 1;
    end

    return p;
end

function plot_total_energy(; total_energy, labels, with_ci=false, ylims=(10, 100), ylabel="P(energy)", plot_mean=true)

    color_list = palette(:tab10);

    p = plot(size=(800, 400), ylims = ylims, xlabel="Time", ylabel=ylabel);

    counter = 1;
    for energy in total_energy

        if length(size(energy)) == 3
            if plot_mean
                mean_energy = mean(energy, dims=(2,3))[:,1,1];
                std_energy = std(energy, dims=(2,3))[:,1,1];
                lower_bound = mapslices(x -> quantile(vec(x), 0.05), energy, dims=[2,3])[:,1,1]
                upper_bound = mapslices(x -> quantile(vec(x), 0.95), energy, dims=[2,3])[:,1,1]
            else
                for path_i in 1:size(energy, 2)
                    for trajectory_i in 1:size(energy, 3)
                        if path_i == 1 && trajectory_i == 1
                            plot!(p, energy[:, path_i, trajectory_i], label=labels[counter], linewidth=2, color=color_list[counter], alpha=0.25)
                        else
                            plot!(p, energy[:, path_i, trajectory_i], label="", linewidth=2, color=color_list[counter], alpha=0.25)
                        end
                    end
                end
            end
        else
            if plot_mean
                mean_energy = mean(energy, dims=2)[:,1];
                # std_energy = std(energy, dims=2)[:,1];
                lower_bound = mapslices(x -> quantile(vec(x), 0.05), energy, dims=2)[:,1]
                upper_bound = mapslices(x -> quantile(vec(x), 0.95), energy, dims=2)[:,1]
            else
                for trajectory_i in 1:size(energy, 2)
                    if trajectory_i == 1
                        plot!(p, energy[:, trajectory_i], label=labels[counter], linewidth=5, color=color_list[counter])
                    else
                        plot!(p, energy[:, trajectory_i], label="", linewidth=5, color=color_list[counter])
                    end
                end
            end
        end

        #     plot!(p, lower_bound, fillrange=upper_bound, alpha=0.2, color=color_list[counter], linewidth=0, label="")
        # end
        if plot_mean
            # lower_bound = mean_energy .- std_energy;
            # upper_bound = mean_energy .+ std_energy;

            plot!(p, mean_energy, label=labels[counter], linewidth=3, color=color_list[counter])
            if with_ci
                plot!(p, lower_bound, fillrange=upper_bound, alpha=0.1, color=color_list[counter], linewidth=0, label="")
            end
        end

        counter += 1;
    end


    return p;
end


function get_energy_spectra(pred)

    num_trajectories = size(pred, 6)
    num_paths = size(pred, 5)
    num_steps = size(pred, 4)

    # Compute energy spectra
    energy_spectrum = []
    for trajectory in 1:num_trajectories
        energy_spectrum_trajectory = []
        for ti = 1:num_steps
            energy_spectrum_ti, k_bins = compute_energy_spectra(pred[:, :, :, ti, :, trajectory]);
            push!(energy_spectrum_trajectory, energy_spectrum_ti);
        end
        energy_spectrum_trajectory = cat(energy_spectrum_trajectory..., dims=3);
        push!(energy_spectrum, energy_spectrum_trajectory);
    end
    energy_spectrum = cat(energy_spectrum..., dims=4); # num_bins, num_paths, num_steps, num_trajectories, 

    return permutedims(energy_spectrum, (1, 3, 2, 4)); # num_bins, num_steps, num_paths, num_trajectories
end

function get_total_energy(pred)

    num_trajectories = size(pred, 6)

    total_energy = []
    for trajectory in 1:num_trajectories
        total_energy_trajectory = compute_total_energy(pred[:, :, :, :, :, trajectory]);
        push!(total_energy, total_energy_trajectory);
    end

    return cat(total_energy..., dims=3); # num_steps, num_paths, num_trajectories
end

function get_mse(pred, _true)
    num_trajectories = size(pred, 6)
    num_paths = size(pred, 5)

    mse = zeros(num_paths, num_trajectories)
    for trajectory in 1:num_trajectories
        for path in 1:num_paths
            mse_trajectory = mean((pred[:, :, :, :, path, trajectory] .- _true[:, :, :, :, trajectory]).^2, dims=(1,2,3,4))[1, 1, 1, 1]
            mse[path, trajectory] = mse_trajectory
        end
    end

    return mse
end

function get_KL_divergence(pred, _true)

    num_trajectories = size(pred, 6)
    num_paths = size(pred, 5)

    kld = zeros(num_paths, num_trajectories)
    for trajectory in 1:num_trajectories
        for path in 1:num_paths
            pred = pred[:, path, trajectory]
            _true = _true[:, trajectory]

            k_pred = kde(pred)
            k_true = kde(_true)
            
            min_val = min(k_pred.x[1], k_true.x[1])
            max_val = max(k_pred.x[end], k_true.x[end])
            
            min_val = k_true.x[1]
            max_val = k_true.x[end]
            
            points = range(min_val, max_val, 100)
            dx = points[2] - points[1]
            p = pdf(k_pred, points)
            q = pdf(k_true, points)
            p[abs.(p) .< 1e-12] .= 1e-12
            q[abs.(q) .< 1e-12] .= 1e-12
            
            kld[path, trajectory] = KLDivergence()(p, q) * dx
        end
    end
    
    return kld

    # k_pred = kde(pred)
    # k_true = kde(_true)

    # min_val = min(k_pred.x[1], k_true.x[1])
    # max_val = max(k_pred.x[end], k_true.x[end])

    # points = range(min_val, max_val, 10000)
    # p = pdf(k_pred, points)
    # q = pdf(k_true, points)
    # p[abs.(p) .< 1e-12] .= 1e-12
    # q[abs.(q) .< 1e-12] .= 1e-12
   
    # return KLDivergence()(p, q)
end



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

results_dir = "results/";
cases_to_plot = [
    "SI_optimized_with_projection_50",
    "acdm_50",
    "refiner_4",
]
labels = [
    "Filtered DNS",
    L"SI_{opt, div}", #"Optimized with Projection",
    "ACDM, 50",
    "Refiner, 4",
]


times_to_plot = [50, 100, 200, 500, 750]
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

save("$(results_dir)npz/true_data.npz", Dict("arr_0" => testset))

for case in cases_to_plot
    pred = load("$results_dir$case.jld2", "data"); # H, W, C, num_steps, nums_paths, num_test


    for time_step in times_to_plot
        preds_to_plot[time_step][case] = sqrt.(pred[:, :, 1, time_step, :, 1].^2 .+ pred[:, :, 2, time_step, :, 1].^2);
    end

    preds_energy_spectra[case] = get_energy_spectra(pred); # num_bins, num_steps, num_paths, num_trajectories
    preds_total_energy[case] = get_total_energy(pred); # num_steps, num_paths, num_trajectories
    pred_mse[case] = get_mse(pred, testset);
    pred_KL_divergence[case] = get_KL_divergence(
        vcat(preds_total_energy[case]...), 
        vcat(true_total_energy...)
    );
    pred_rate_of_change_magnitude[case] = mean(abs.(pred[:, :, :, 2:end, :, :] .- pred[:, :, :, 1:end-1, :, :])./dt, dims=(1,2,3))[1, 1, 1, :, :, :];

    println("Loaded $case")
end

figure_folder = "comparison_figures_for_paper";

for time_step = times_to_plot
    plot_list = (preds_energy_spectra[case][:, time_step, :, :] for case in cases_to_plot);
    plot_list = (true_energy_spectra[:, time_step, :], plot_list..., );
    plot_energy_spectra(;
        spectra=plot_list, 
        labels=labels, 
        with_ci=true
    )
    savefig("$figure_folder/energy_spectra_$(case)time_step_$(time_step).pdf")
end
    
plot_list = (preds_total_energy[case] for case in cases_to_plot);
plot_list = (true_total_energy, plot_list..., );
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels, 
    with_ci=true,
    ylabel="Total Energy",
    plot_mean=true
)
savefig("$figure_folder/total_energy_$(case).pdf")
    
plot_list = (vcat(energy...) for energy in plot_list);
plot_energy_density(;
    energies=plot_list, 
    labels=labels, 
)
savefig("$figure_folder/energy_density_$(case).pdf")
    
    
plot_list = (pred_rate_of_change_magnitude[case] for case in cases_to_plot);
plot_list = (true_rate_of_change_magnitude, plot_list..., );
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels, 
    with_ci=true,
    ylims=(0, 15),
    ylabel="Rate of change",
    plot_mean=true
)
savefig("$figure_folder/rate_of_change_magnitude_$(case).pdf")

for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = [4]
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        println("MSE for $case$num_generator_steps: ", round(mean(pred_mse["$case$num_generator_steps"]), digits=3))
    end
end
for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = [4]
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        println("MSE for $case$num_generator_steps: ", round(std(pred_mse["$case$num_generator_steps"]), digits=3))
    end
end


for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = [4]
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        println("Mean KL Divergence for $case$num_generator_steps: ", round(mean(pred_KL_divergence["$case$num_generator_steps"]), digits=3))
    end
end

for case in cases_to_plot
    if case == "refiner_"
        _num_generator_steps_list = [4]
    else
        _num_generator_steps_list = num_generator_steps_list;
    end
    for num_generator_steps in _num_generator_steps_list
        println("Std KL Divergence for $case$num_generator_steps: ", round(std(pred_KL_divergence["$case$num_generator_steps"]), digits=3))
    end
end


velocity_magnitude_true = sqrt.(testset[:, :, 1, :, 1].^2 .+ testset[:, :, 2, :, 1].^2);
for time_step in times_to_plot
    heatmap(velocity_magnitude_true[:, :, time_step], aspect_ratio=:equal, legend=false, xticks=false, yticks=false, clim=(minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)), color=cgrad(:Spectral_11, rev=true))
    savefig("$figure_folder/velocity_magnitude_true_time_step_$(time_step).pdf")
end

for time_step in times_to_plot
    for case in cases_to_plot
        if case == "refiner_"
            _num_generator_steps_list = [4]
        else
            _num_generator_steps_list = num_generator_steps_list
        end
        for num_generator_steps in _num_generator_steps_list
            for path_i in range(1, 5)
                velocity_magnitude = preds_to_plot[time_step]["$case$num_generator_steps"][:, :, path_i]
                heatmap(velocity_magnitude, aspect_ratio=:equal, legend=false, xticks=false, yticks=false, clim=(minimum(velocity_magnitude_true), maximum(velocity_magnitude_true)), color=cgrad(:Spectral_11, rev=true))
                savefig("$figure_folder/velocity_magnitude_$(case)$(num_generator_steps)_time_step_$(time_step)_path_$(path_i).pdf")
            end
        end
    end
end

for case in cases_to_plot

    if case == "refiner_"
        _num_generator_steps_list = [4]
    else
        _num_generator_steps_list = [10, 25, 50];
    end

    for num_generator_steps in _num_generator_steps_list
        pred = load("$results_dir$case$num_generator_steps.jld2", "data"); # H, W, C, num_steps, nums_paths, num_test
        pred_vel_magnitude = sqrt.(pred[:, :, 1, :, :, 1].^2 .+ pred[:, :, 2, :, :, 1].^2);
        plot_list = [pred_vel_magnitude[:, :, :, i] for i in 1:5];
        create_gif(plot_list, "animations/$case$num_generator_steps.mp4", ["", "", "", "", ""])
    end
end




time_step = 750
num_generator_steps = 50
plot_list = (preds_energy_spectra["$case$num_generator_steps"][:, time_step, :, :] for case in cases_to_plot);
plot_list = (true_energy_spectra[:, time_step, :], plot_list..., );
plot_energy_spectra(;
    spectra=plot_list, 
    labels=labels, 
    with_ci=false
)
savefig("energy_spectra_gen_steps_$(num_generator_steps)_time_step_$(time_step).pdf")

plot_list = (preds_total_energy["$case$num_generator_steps"] for case in cases_to_plot);
plot_list = (true_total_energy, plot_list..., );
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels, 
    with_ci=false,
    ylabel="Total Energy",
    plot_mean=true
)
savefig("total_energy_gen_steps_$(num_generator_steps).pdf")

plot_list = (vcat(energy...) for energy in plot_list);
plot_energy_density(;
    energies=plot_list, 
    labels=labels, 
)
savefig("energy_density_gen_steps_$(num_generator_steps).pdf")


plot_list = (pred_rate_of_change_magnitude["$case$num_generator_steps"] for case in cases_to_plot);
plot_list = (true_rate_of_change_magnitude, plot_list..., );
plot_total_energy(;
    total_energy=plot_list, 
    labels=labels, 
    with_ci=false,
    ylims=(1000, 2500),
    ylabel="Rate of change",
    plot_mean=false
)
savefig("rate_of_change_magnitude_gen_steps_$num_generator_steps.pdf")







# pred_no_projection = permutedims(pred_no_projection, (2, 1, 3, 4, 5, 6));
test_vel_magnitude = sqrt.(testset[:, :, 1, :, 1].^2 .+ testset[:, :, 2, :, 1].^2);
pred_vel_magnitude = sqrt.(pred_no_projection[:, :, 1, :, 1, 1].^2 .+ pred_no_projection[:, :, 2, :, 1, 1].^2);








# pred = load("results/SI_not_optimized_50.jld2", "data"); # H, W, C, num_steps, nums_paths, num_test
# _true = testset;


# H, W, C, num_steps, nums_paths, num_trajectories = size(pred)
    
# time_step = 100

# pred_mean = mean(pred[:, :, :, time_step, :, :], dims=(1, 2))[1, 1, :, :, :]
# true_mean = mean(_true[:, :, :, time_step, :], dims=(1, 2))[1, 1, :, :]
# pred_std = std(pred[:, :, :, time_step, :, :], dims=(1, 2), corrected=false)[1, 1, :, :, :]
# true_std = std(_true[:, :, :, time_step, :, :], dims=(1, 2), corrected=false)[1, 1, :, :]

# corr_trajectories = []
# for trajectory_i in 1:num_trajectories
#     corr_path = []
#     for path_i in 1:nums_paths
#         corr_path_i = mean((pred[:, :, :, time_step, path_i, trajectory_i] .- pred_mean[:, path_i, trajectory_i]) .* (_true[:, :, :, time_step, trajectory_i] .- true_mean[:, trajectory_i])) ./ 
#             max.(pred_std[:, path_i, trajectory_i] .* true_std[:, trajectory_i], floatmin(Float32))
#         push!(corr_path, corr_path_i)
#         println(corr_path_i)
#     end
#     corr_path = stack(corr_path, dims=1)
#     push!(corr_trajectories, corr_path)
# end
# corr_trajectories[1]

# corr = dropdims(corr, dims=1) # Shape (T, B)

# corr = mean(corr, dims=2) # Reduce over batch
# corr = dropdims(corr, dims=2) # Shape (T)



