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
using CUDA
using Plots

using ForwardDiff

using Statistics
using Plots

interpolant(coefs, dev=gpu_device()) = begin

    num_total_coefs = length(coefs)
    coefs = reshape(coefs, 2, num_total_coefs ÷ 2)

    alpha = t -> get_alpha_series(t, coefs[1, :])
    beta = t -> get_beta_series(t, coefs[2, :])
    # gamma = t -> get_gamma_series(t, coefs[3, :] |> dev)

    dalpha_dt = t -> get_dalpha_series_dt(t, coefs[1, :])
    dbeta_dt = t -> get_dbeta_series_dt(t, coefs[2, :])
    # dgamma_dt = t -> get_dgamma_series_dt(t, coefs[3, :] |> dev)

    gamma = t -> 0.1 .*(1f0 .- t) * 10
    dgamma_dt = t -> -0.1 .* 1f0 * 10

    return Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt) 
end

test_case = "kolmogorov";
# test_case = "incompressible_flow";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
# mask = mask |> dev;
num_train = size(trainset, 5);
num_steps = size(trainset, 4);
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3);



if test_case == "kolmogorov"
    omega = [0.04908f0, 0.04908f0]
elseif test_case == "incompressible_flow"
    omega = [0.03125f0, 0.03125f0]
end


# 0.1 * gamma
coefs = [
    -1.05734  -0.00348673  -0.0312818  -0.00382112  -0.00580364;
    -1.05611   0.00127347  -0.0293777   0.00343358  -0.00645624
]

num_steps = 250
t_vec = LinRange(0f0, 1f0, num_steps)
dt = 1f0 / num_steps

t_vec = LinRange(0f0, 1f0, num_steps)
t_all = reshape(t_vec, 1, 1, 1, num_steps)

energy_good = zeros(num_steps, 10*44);


inter = interpolant(coefs);
for i in 1:44
    for j in 1:10
        idx = rand(1:num_steps-1)
        x0 = trainset[:, :, :, idx:idx, i]
        x1 = trainset[:, :, :, (idx+1):(idx+1), i]
        noise = randn(size(x0)[1:3]..., 1);
        noise = repeat(noise, outer = [1, 1, 1, num_steps]);
        test_I = inter.interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* inter.gamma(t_all);
        test_I = reshape(test_I, size(test_I)[1:4]..., 1);
        energy_new = compute_total_energy(test_I, omega);
        energy_good[:, (i-1)*10 + j] = energy_new;
    end
end

energy_good_flat = reshape(energy_good, num_steps*10*44)

histogram!(
    energy_good_flat, bins=50, label="Histogram SI opt x10", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true, color=:red, alpha=0.5
)

energy_bad = zeros(num_steps, 10*44);


inter = get_interpolant("linear", "quadratic", "linear", 0.1);
for i in 1:44
    for j in 1:10
        idx = rand(1:num_steps-1)
        x0 = trainset[:, :, :, idx:idx, i]
        x1 = trainset[:, :, :, (idx+1):(idx+1), i]
        noise = randn(size(x0)[1:3]..., 1);
        noise = repeat(noise, outer = [1, 1, 1, num_steps]);
        test_I = inter.interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* inter.gamma(t_all);
        test_I = reshape(test_I, size(test_I)[1:4]..., 1);
        energy_new = compute_total_energy(test_I, omega);
        energy_bad[:, (i-1)*10 + j] = energy_new;
    end
end


energy_bad_flat = reshape(energy_bad, num_steps*10*44)



histogram!(
    energy_bad_flat, bins=50, label="Histogram SI default", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true, color=:green, alpha=0.30
)
savefig("kolmogorov_energy.png")

show()