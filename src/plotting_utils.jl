using Plots
# using WGLMakie
using GLMakie
using KernelDensity

# function create_gif(A, filename, plot_titles)
    
#     p_list = [];
#     for i = 1:length(A)
#         p = heatmap(
#             A[i][:,:,1], legend=false, xticks=false, yticks=false, 
#             # clim=(minimum(A[1]), maximum(A[1])), #clim=(0.15, 0.75),
#             aspect_ratio=:equal,  title=plot_titles[i], color=cgrad(:Spectral_11, rev=true),
#         )
#         push!(p_list, p)
#     end

#     if length(A) > 2 && length(A) % 2 == 0
#         p = plot(p_list..., layout=(2, length(A) ÷ 2), size=(1600, 800))

#     elseif length(A) == 1
#         p = plot(p_list..., layout=(1, 1), size=(800, 400))
#     else
#         p = plot(p_list..., layout=(1, length(A)), size=(length(A)*400, 400))
#     end

#     anim = @animate for i=1:size(A[1], 3)
#         for j = 1:length(A)
#             p[j][1][:z] = transpose(A[j][:,:,i])
#         end
#     end
#     gif(anim, filename, fps = 15)
    
# end


function create_gif(A, filename, plot_titles)
    A_obs = map(A) do A
        GLMakie.Observable(A[:, :, 1])
    end

    # Calculate figure size based on number of plots
    n_plots = length(A)
    fig_width = 400 * n_plots # 200 pixels per plot
    fig_height = 400 # Keep height constant
    fig = Figure(; size = (fig_width, fig_height))
    for (i, A_obs) in enumerate(A_obs)
        ax = Axis(
            fig[1, i];
            title = plot_titles[i],
            aspect = DataAspect(),
            xticksvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false,
        )
        GLMakie.heatmap!(ax, A_obs; colormap = Reverse(:Spectral_11))
    end
    ntime = size(A[1], 3)
    stream = GLMakie.VideoStream(fig; framerate = 15)
    for i = 1:ntime
        for (A_obs, A) in zip(A_obs, A)
            A_obs[] = A[:, :, i]
        end
        GLMakie.recordframe!(stream)
    end
    save("$filename.mp4", stream)
end

function plot_energy_density(; energies, labels)

    color_list = palette(:tab10);

    x = LinRange(0, 100, 2048)
    p = Plots.plot(x_lims=(0, 100), xlabel="Energy", ylabel="P(energy)", normed=true)
    counter = 1
    for energy in energies
        k = kde(energy)
        ik = InterpKDE(k)
        z = pdf(ik, x)
        Plots.plot!(p, x, z, label=labels[counter], linewidth=3, color=color_list[counter])
        Plots.plot!(p, x, z, fillrange=0, alpha=0.2, color=color_list[counter], linewidth=0, label="")
        counter += 1
    end
    return p
end

function plot_energy_spectra(; spectra, labels, with_ci=false, ylims=(1e-8, 5000))

    color_list = palette(:tab10);

    p = Plots.plot(ylims = ylims, xaxis=:log, yaxis=:log, xlabel="Frequency", ylabel="Energy");

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

        Plots.plot!(p, mean_spectrum, label=labels[counter], linewidth=3, color=color_list[counter])
        # plot!(p, lower_bound, alpha=0.2, color=color_list[counter], linewidth=5, label="")
        if with_ci
            Plots.plot!(p, lower_bound, fillrange=upper_bound, alpha=0.2, color=color_list[counter], linewidth=0, label="")
        end

        counter += 1;
    end

    return p;
end

function plot_total_energy(; total_energy, labels, with_ci=false, ylims=(10, 100), ylabel="P(energy)", plot_mean=true)

    color_list = palette(:tab10);

    p = Plots.plot(size=(800, 400), ylims = ylims, xlabel="Time", ylabel=ylabel);

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
                            Plots.plot!(p, energy[:, path_i, trajectory_i], label=labels[counter], linewidth=2, color=color_list[counter], alpha=0.25)
                        else
                            Plots.plot!(p, energy[:, path_i, trajectory_i], label="", linewidth=2, color=color_list[counter], alpha=0.25)
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
                        Plots.plot!(p, energy[:, trajectory_i], label=labels[counter], linewidth=5, color=color_list[counter])
                    else
                        Plots.plot!(p, energy[:, trajectory_i], label="", linewidth=5, color=color_list[counter])
                    end
                end
            end
        end

        #     plot!(p, lower_bound, fillrange=upper_bound, alpha=0.2, color=color_list[counter], linewidth=0, label="")
        # end
        if plot_mean
            # lower_bound = mean_energy .- std_energy;
            # upper_bound = mean_energy .+ std_energy;

            Plots.plot!(p, mean_energy, label=labels[counter], linewidth=3, color=color_list[counter])
            if with_ci
                Plots.plot!(p, lower_bound, fillrange=upper_bound, alpha=0.1, color=color_list[counter], linewidth=0, label="")
            end
        end

        counter += 1;
    end

    return p;
end