
using JLD2
using StochasticInterpolants
using Plots
using KernelDensity

function get_density(energy)
    k = kde(energy)
    ik = InterpKDE(k)

    x = LinRange(20, 60, 2048)
    z = pdf(ik, x)
    return z, x
end

pred_sol = load("pred_sol_not_optimized.jld2", "data");
# pred_sol = load("pred_sol.jld2", "data");
pred_true = load("testset.jld2", "data");

H, W, C, num_steps, num_test_paths, num_test_trajectories = size(pred_sol)


energy_pred = compute_total_energy(reshape(pred_sol, H, W, C, num_steps, num_test_paths*num_test_trajectories), [0.04908f0, 0.04908f0]);

# Remove rows with values larger than 100
energy_pred_modified = []
for i in 1:num_test_paths*num_test_trajectories
    if maximum(energy_pred[:, i]) < 100
        push!(energy_pred_modified, energy_pred[:, i])
    end
end
energy_pred_modified = hcat(energy_pred_modified...)

energy_true = compute_total_energy(pred_true, [0.04908f0, 0.04908f0]);

t_vec = 1:num_steps;
plot(t_vec, energy_true, linewidth=3, color=:blue, alpha=0.35, legend = false, ylims=(20, 60))
# plot!(t_vec, energy_pred_modified, linewidth=3, color=:green, alpha=0.35, legend = false)
savefig("energy_time_true_only.pdf")

### Energy distributions ###
energy_true_flat = reshape(energy_true, num_steps*num_test_trajectories);
energy_pred_flat = reshape(energy_pred_modified, num_steps*size(energy_pred_modified, 2)*size(energy_pred_modified, 3));
energy_true_density, x = get_density(energy_true_flat)
energy_pred_density, x = get_density(energy_pred_flat)

energy_true_density, x = get_density(energy_true[end, :]);
energy_pred_density, x = get_density(energy_pred_modified[end, :]);

plot(energy_true_density, x, ylabel="Q(u)", legend=false, linewidth=3, color=palette(:tab10)[1],  ylims=(20, 60), xlims=(0, 0.1))
plot!(energy_true_density, x, fillrange=zero(x), ylabel="Q(u)", legend=false, linewidth=0, alpha=0.25, color=palette(:tab10)[1], xlims=(0, 0.1))
# plot!(energy_pred_density, x, xlabel="t", ylabel="Q(u)", legend=false, linewidth=3, color=palette(:tab10)[3])
# plot!(energy_pred_density, x, fillrange=zero(x), xlabel="t", ylabel="Q(u)", legend=false, linewidth=0, alpha=0.25, color=palette(:tab10)[3])
plot!(size=(400,400))
savefig("energy_density_true_only.pdf")


### Plot velocity field ###
time_step = 10;
trajectory = 1;

velocity_magnitude_true = sqrt.(pred_true[:, :, 1, :, :].^2 + pred_true[:, :, 2, :, :].^2);
velocity_magnitude_pred = sqrt.(pred_sol[:, :, 1, :, :, :].^2 + pred_sol[:, :, 2, :, :, :].^2);

heatmap(velocity_magnitude_true[:, :, time_step, trajectory], color=:viridis, legend=false, colorbar=true, zlims=(3.0, 0.0))
plot!(size=(500,400))
savefig("velocity_different_1.pdf")

heatmap(velocity_magnitude_true[:, :, time_step, trajectory+1], color=:viridis, legend=false, colorbar=true, zlims=(3.0, 0.0))
plot!(size=(500,400))
savefig("velocity_different_2.pdf")