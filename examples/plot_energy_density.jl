ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "20GiB"
ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"

using StochasticInterpolants
using Lux
using YAML
using Random
using NPZ
using LuxCUDA
using Optimisers
using FileIO

using Statistics
using Plots

using KernelDensity
using Interpolations
using StatsPlots


# base_dir = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/data/kolmogorov_64"
# files = readdir(base_dir, sort = true)
# for (i, file) in enumerate(files)
#     new_file = joinpath(base_dir, "sim_$(i).jld2")
#     file = joinpath(base_dir, file)
#     mv(file, new_file)
# end



test_case = "kolmogorov";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
omega = [0.04908f0, 0.04908f0]
energy = compute_total_energy(trainset, omega)

t_vec = LinRange(0.0, 80.0, 3200)
# t_vec = reshape(t_vec, 3200, 1)
# t_vec = repeat(t_vec, 1, 45)

i = 10
k = kde(energy[i, :])
ik = InterpKDE(k)
z = pdf(ik, energy[i, :])

true_energy_flat = reshape(energy, 250*45)
histogram(
    true_energy_flat, bins=50, label="Histogram True", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true
)
plot!(k.x, k.density, label="KDE", linewidth=2, color=:red)

dt = 5e-4
timesteps = 3200
ZZ = zeros(timesteps, 2048)
XX = zeros(timesteps, 2048)
for i = 1:timesteps
    k = kde(energy[i, :])
    # z = k.density
    # x = k.x
    # ZZ[i, :] = z
    # XX[i, :] = x

    x = LinRange(7.5e3, 3e4, 2048)
    ik = InterpKDE(k)

    z = pdf(ik, x)
    ZZ[i, :] = z
    XX[i, :] = x
    
end
heatmap(transpose(ZZ), c = :vik, xlabel="Time", ylabel="Energy", title="Density of Energy",
    yticks=(LinRange(1, 2048, 5), LinRange(7.5e3, 3e4, 5)),
    xticks=(LinRange(1, timesteps, 5), LinRange(0, timesteps*50*dt, 5))
)
savefig("kolmogorov_energy_density.png")

plot(LinRange(0, timesteps*50*dt, timesteps),energy[1:timesteps, :], xlabel="Time", ylabel="Energy", title="Total Energy of the System", legend=false)
plot!(LinRange(0, timesteps*50*dt, timesteps),
    mean(energy[1:timesteps, :], dims=2), linewidth=10, xlabel="Time", color=:black,
ylabel="Energy", title="Total Energy of the System", label="Mean Energy")
savefig("kolmogorov_energy.png")

plot(LinRange(7.5e3, 3e4, 2048), ZZ[end, :], linewidth=5,
xlabel="Energy", ylabel="Density", title="Density of Energy at t=$(timesteps*50*dt)", legend=false)
savefig("kolmogorov_energy_density_end.png")


plot(LinRange(7.5e3, 3e4, 2048), ZZ[1, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(1*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[250, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(250*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[500, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(500*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[1000, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(1000*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[1500, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(1500*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[2000, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(2000*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[2500, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(2500*50*dt)")
plot!(LinRange(7.5e3, 3e4, 2048), ZZ[3000, :], linewidth=3,
xlabel="Energy", ylabel="Density", label="t=$(3000*50*dt)")
savefig("kolmogorov_energy_density_time.png")




k = kde(energy[i, :])
z = k.density
x = k.x
ZZ[i, :] = z
XX[i, :] = x

ik = InterpKDE(k)

z = pdf(ik, x)

plot(XX[1, :], ZZ[1,:], xlabel="Energy", ylabel="Density", title="Density of Energy", legend=false)
plot!(XX[1000, :], ZZ[1000,:], xlabel="Energy", ylabel="Density", title="Density of Energy", legend=false)
plot!(XX[2000, :], ZZ[2000,:], xlabel="Energy", ylabel="Density", title="Density of Energy", legend=false)
plot!(XX[3000, :], ZZ[3000,:], xlabel="Energy", ylabel="Density", title="Density of Energy", legend=false)

contourf(t_vec, XX, ZZ)

t_vec = repeat(t_vec, 1, 2048)

heatmap(t_vec, XX, ZZ)

# trainset = load("data/kolmogorov/sim_2.jld2", "u")
trainset = reshape(trainset, 130, 130, 2, 1201, 1)
using Plots

x = sqrt.(trainset[:, :, 1, 1:10:end].^2 + trainset[:, :, 2, 1:10:end].^2)
create_gif(
    (x, ), 
    "kolmogorov.gif", 
    ("1")
)


xx =[]
for i = 1:8
    data_train = load("data/kolmogorov/sim_$i.jld2")
    u = data_train["u"]

    u = 0.5 .* (u[1:end-1, :, :, :] + u[2:end, :, :, :])
    u = 0.5 .* (u[:, 1:end-1, :, :] + u[:, 2:end, :, :])

    x = sqrt.(u[:, :, 1, 1:3:1500].^2 + u[:, :, 2, 1:3:1500].^2)

    push!(xx, x)
end
plot(energy, label="Energy", xlabel="Time", ylabel="Energy", title="Total Energy of the System")


create_gif(
    xx, 
    "kolmogorov.mp4", 
    ("1", "2", "3", "4", "5", "6", "7", "8")
)

# Create a gif
x1, p1 = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2), trainset_pars[1, 1, 1];
x2, p2 = sqrt.(trainset[:, :, 1, :, 2].^2 + trainset[:, :, 2, :, 2].^2), trainset_pars[1, 1, 2];
x3, p3 = sqrt.(trainset[:, :, 1, :, 3].^2 + trainset[:, :, 2, :, 3].^2), trainset_pars[1, 1, 3];
x4, p4 = sqrt.(trainset[:, :, 1, :, 4].^2 + trainset[:, :, 2, :, 4].^2), trainset_pars[1, 1, 4];
x5, p5 = sqrt.(trainset[:, :, 1, :, 5].^2 + trainset[:, :, 2, :, 5].^2), trainset_pars[1, 1, 5];
x6, p6 = sqrt.(trainset[:, :, 1, :, 6].^2 + trainset[:, :, 2, :, 6].^2), trainset_pars[1, 1, 6];
x7, p7 = sqrt.(trainset[:, :, 1, :, 7].^2 + trainset[:, :, 2, :, 7].^2), trainset_pars[1, 1, 7];
x8, p8 = sqrt.(trainset[:, :, 1, :, 8].^2 + trainset[:, :, 2, :, 8].^2), trainset_pars[1, 1, 8];

create_gif(
    (x1, x2, x3, x4, x5, x6, x7, x8), 
    "HF.gif", 
    ("Ma = $p1", "Ma = $p2", "Ma = $p3", "Ma = $p4", "Ma = $p5", "Ma = $p6", "Ma = $p7", "Ma = $p8")
)


