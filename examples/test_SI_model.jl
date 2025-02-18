ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "10GiB"
ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"

using JLD2
using StochasticInterpolants
using Lux
using YAML
using Random
using NPZ
using LuxCUDA
using Optimisers
using FileIO
using BenchmarkTools
using Plots

CUDA.reclaim()
GC.gc()

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();


# Choose between "transonic_cylinder_flow", "incompressible_flow", "turbulence_in_periodic_box"
test_case = "incompressible_flow";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
mask = mask |> dev;

num_train = size(testset, 5);
num_steps = size(testset, 4);
H, W, C = size(testset, 1), size(testset, 2), size(testset, 3);


##### Hyperparameters #####
model_base_dir = "trained_models/";
# model_name = "forecasting_model_optimized_project_structure";
model_name = "forecasting_model_optimized_project_new";
# model_name = "forecasting_model_not_optimized";

checkpoint_manager = CheckpointManager(
    test_case, model_name; base_folder=model_base_dir
);

config = checkpoint_manager.neural_network_config

##### Forecasting SI model #####
# Define the velocity model


if occursin("structure", model_name)
    velocity = PhysicsConsistentModel(
        (H, W),
        config["model_args"]
    );
else
    velocity = get_SI_neural_network(;
        image_size=(H, W),
        model_params=config["model_args"]
    );
end;

# Get Interpolant
interpolant = get_interpolant(
    config["interpolant_args"]["alpha"],
    config["interpolant_args"]["beta"],
    config["interpolant_args"]["gamma"],
    config["interpolant_args"]["gamma_multiplier"] |> Float32,
);

if model_name == "forecasting_model_optimized" || model_name == "forecasting_model_optimized_new"  || model_name == "forecasting_model_optimized_refactored" || model_name == "forecasting_model_optimized_project_new"
    print("Using optimized coefficients")

    coefs = [
        -1.05734  -0.00348673  -0.0312818  -0.00382112  -0.00580364;
        -1.05611   0.00127347  -0.0293777   0.00343358  -0.00645624
    ]
    coefs = coefs .|> Float32;
    
    interpolant = Interpolant(
        t -> get_alpha_series(t, coefs[1, :]), 
        t -> get_beta_series(t, coefs[2, :]), 
        t -> get_dalpha_series_dt(t, coefs[1, :]),
        t -> get_dbeta_series_dt(t, coefs[2, :]), 
        t -> 0.1f0 .* (1f0 .- t),
        t -> -1f0 .* 0.1f0
    )
end;
interpolant
# Get diffusion coefficient
diffusion_coefficient = get_diffusion_coefficient(
    config["diffusion_args"]["type"],
    config["diffusion_args"]["multiplier"] |> Float32,
);

if config["model_args"]["projection"] == "divergence_free"
    projection = project_onto_divergence_free;
elseif config["model_args"]["projection"] == "projection_with_obstacle"
    projection = projection_with_obstacle;
else
    projection = nothing;
end;

# Initialise the SI model
model = FollmerStochasticInterpolant(
    velocity; 
    interpolant=interpolant,
    diffusion_coefficient=diffusion_coefficient,
    projection=projection,
    len_history=config["model_args"]["len_history"],
    dev=dev
);

weights_and_states = checkpoint_manager.load_model();
ps = weights_and_states.ps |> dev;
st = weights_and_states.st |> dev;

##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
num_test_trajectories = size(testset, 5);
num_test_paths = 5;
num_generator_steps = 100



omega = [0.04908f0, 0.04908f0]

st_ = Lux.testmode(st);
num_test_trajectories = size(testset, 5);
num_test_paths = 5;
energy_spectrum_true, k_bins = compute_energy_spectra(testset[:, :, :, end, :]);

for num_generator_steps = [100]
    pred_sol = zeros(H, W, C, num_steps, num_test_paths, num_test_trajectories);
    energy_pred = zeros(num_steps, num_test_paths, num_test_trajectories);
    energy_spectrum_pred = zeros(size(k_bins,1), num_test_paths, num_test_trajectories);

    for i in 1:num_test_trajectories
        sol = compute_multiple_SDE_steps(
            init_condition=testset[:, :, :, 1:config["model_args"]["len_history"], i],
            parameters=testset_pars[:, 1:1, i],
            num_physical_steps=size(testset, 4),
            num_generator_steps=num_generator_steps,
            num_paths=num_test_paths,
            model=model,
            ps=ps,
            st=st_,
            rng=rng,
            dev=gpu_device(),
            mask=nothing,
        )
        pred_sol[:, :, :, :, :, i] = sol;
        # energy_pred[:, :, i] = compute_total_energy(sol, omega);
        # energy_spectrum_pred[:, :, i], _ = compute_energy_spectra(sol[:, :, :, end, :]);
    end;
    save("$(model_name)$(num_generator_steps).jld2", "data", pred_sol)

end;










projection = project_onto_divergence_free;

# Initialise the SI model
model = FollmerStochasticInterpolant(
    velocity; 
    interpolant=interpolant,
    diffusion_coefficient=diffusion_coefficient,
    projection=projection,
    len_history=config["model_args"]["len_history"],
    dev=dev
);

weights_and_states = checkpoint_manager.load_model();
ps = weights_and_states.ps |> dev;
st = weights_and_states.st |> dev;

##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
num_test_trajectories = size(testset, 5);
num_test_paths = 5;
num_generator_steps = 100



omega = [0.04908f0, 0.04908f0]

st_ = Lux.testmode(st);
num_test_trajectories = size(testset, 5);
num_test_paths = 5;
energy_spectrum_true, k_bins = compute_energy_spectra(testset[:, :, :, end, :]);

for num_generator_steps = [10, 25, 50, 100]
    pred_sol = zeros(H, W, C, num_steps, num_test_paths, num_test_trajectories);
    energy_pred = zeros(num_steps, num_test_paths, num_test_trajectories);
    energy_spectrum_pred = zeros(size(k_bins,1), num_test_paths, num_test_trajectories);

    for i in 1:num_test_trajectories
        sol = compute_multiple_SDE_steps(
            init_condition=testset[:, :, :, 1:config["model_args"]["len_history"], i],
            parameters=testset_pars[:, 1:1, i],
            num_physical_steps=size(testset, 4),
            num_generator_steps=num_generator_steps,
            num_paths=num_test_paths,
            model=model,
            ps=ps,
            st=st_,
            rng=rng,
            dev=gpu_device(),
            mask=nothing,
        )
        pred_sol[:, :, :, :, :, i] = sol;
        # energy_pred[:, :, i] = compute_total_energy(sol, omega);
        # energy_spectrum_pred[:, :, i], _ = compute_energy_spectra(sol[:, :, :, end, :]);
    end;
    save("$(model_name)_with_projection_$(num_generator_steps).jld2", "data", pred_sol)

end;



























for num_generator_steps = [10, 25, 50, 100]
    pred_sol = zeros(H, W, C, num_steps, num_test_paths, num_test_trajectories);
    energy_pred = zeros(num_steps, num_test_paths, num_test_trajectories);
    energy_spectrum_pred = zeros(size(k_bins,1), num_test_paths, num_test_trajectories);
    for i in 1:num_test_trajectories
        sol = compute_multiple_SDE_steps(
            init_condition=testset[:, :, :, 1:config["model_args"]["len_history"], i],
            parameters=testset_pars[:, 1:1, i],
            num_physical_steps=size(testset, 4),
            num_generator_steps=num_generator_steps,
            num_paths=num_test_paths,
            model=model,
            ps=ps,
            st=st_,
            rng=rng,
            dev=gpu_device(),
            mask=nothing,
        )
        pred_sol[:, :, :, :, :, i] = sol;
        # energy_pred[:, :, i] = compute_total_energy(sol, omega);
        # energy_spectrum_pred[:, :, i], _ = compute_energy_spectra(sol[:, :, :, end, :]);
    end;
    save("$(model_name)_no_projection_$(num_generator_steps).jld2", "data", pred_sol)

end;

energy_true = compute_total_energy(testset, omega);

energy_true_flat = reshape(energy_true, num_steps*num_test_trajectories);
energy_pred_flat = reshape(energy_pred, num_steps*num_test_paths*num_test_trajectories);

energy_true = compute_total_energy(testset, omega);

energy_true_flat = reshape(energy_true, num_steps*num_test_trajectories);
energy_pred_flat = reshape(energy_pred, num_steps*num_test_paths*num_test_trajectories);

histogram(
    energy_true_flat, bins=50, label="Histogram True", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true
)
histogram!(
    energy_pred_flat[1:1000], bins=50, label="Histogram Pred", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true, alpha=0.5
)

t_vec = 1:num_steps;
plot(t_vec, energy_true, linewidth=3, color=:blue)
plot!(t_vec, reshape(energy_pred, num_steps, num_test_paths*num_test_trajectories), linewidth=3, color=:green)


save("testset.jld2", "data", testset)


@time compare_sde_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    num_test_paths,
    normalize_data,
    mask,
    num_generator_steps,
    "$(model_name)_$(num_generator_steps)",
    rng,
    dev,
)




using Plots
histogram(
    energy_true_flat, bins=50, label="Histogram True", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true
)
histogram!(
    energy_pred_flat[1:5000], bins=50, label="Histogram Pred", 
    xlabel="Energy", ylabel="Frequency", 
    title="Histogram of Energy", normed=true, alpha=0.5
)




t_vec = 1:num_steps;
plot(t_vec, energy_true, linewidth=3, color=:blue)
plot!(t_vec, reshape(energy_pred, num_steps, num_test_paths*num_test_trajectories), linewidth=3, color=:green)






energy_spectra_pred_mean = mean(reshape(energy_spectrum_pred, size(k_bins,1), num_test_paths * num_test_trajectories), dims=2);
energy_spectra_pred_std = std(reshape(energy_spectrum_pred, size(k_bins,1), num_test_paths * num_test_trajectories), dims=2);

energy_spectra_true_mean = mean(energy_spectrum_true, dims=2);
energy_spectra_true_std = std(energy_spectrum_true, dims=2);

plot(k_bins, abs.(energy_spectra_true_mean .- energy_spectra_true_std), fillrange=energy_spectra_true_mean .+ energy_spectra_true_std, color=:blue, alpha=0.25, xaxis=:log, yaxis=:log, primary=false)
plot!(k_bins, abs.(energy_spectra_pred_mean .- energy_spectra_pred_std), fillrange=energy_spectra_pred_mean .+ energy_spectra_pred_std, color=:green, alpha=0.25, xaxis=:log, yaxis=:log, primary=false)
plot!(k_bins, energy_spectra_true_mean, color=:blue, label="True", linewidth=3, xaxis=:log, yaxis=:log)
plot!(k_bins, energy_spectra_pred_mean, color=:green, label="Stochastic Interpolant", linewidth=5, xaxis=:log, yaxis=:log)


@time compare_sde_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    num_test_paths,
    normalize_data,
    mask,
    num_generator_steps,
    "lol",
    rng,
    dev,
)























for diffusion_multiplier = [0.1f0];
    num_generator_steps = 100

    # Get Interpolant
    interpolant = get_interpolant(
        config["interpolant_args"]["alpha"],
        config["interpolant_args"]["beta"],
        config["interpolant_args"]["gamma"],
        T(config["interpolant_args"]["gamma_multiplier"]),
    );

    coefs = [
        -1.05734  -0.00348673  -0.0312818  -0.00382112  -0.00580364;
        -1.05611   0.00127347  -0.0293777   0.00343358  -0.00645624
    ]
    coefs = coefs .|> T;

    interpolant = Interpolant(
        t -> get_alpha_series(t, coefs[1, :]), 
        t -> get_beta_series(t, coefs[2, :]), 
        t -> get_dalpha_series_dt(t, coefs[1, :]),
        t -> get_dbeta_series_dt(t, coefs[2, :]), 
        t -> 0.1f0 .* (1f0 .- t),
        t -> -1f0 .* 0.1f0
    )

    # Get diffusion coefficient
    diffusion_coefficient = get_diffusion_coefficient(
        "follmer_optimal", #config["diffusion_args"]["type"],
        T(config["diffusion_args"]["multiplier"]),
    );

    if config["model_args"]["projection"] == "divergence_free"
        projection = project_onto_divergence_free;
    else
        projection = nothing;
    end;

    # Initialise the SI model
    model = FollmerStochasticInterpolant(
        velocity; 
        # interpolant=interpolant, #Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
        interpolant=interpolant,
        diffusion_coefficient=diffusion_coefficient, #t -> get_gamma_series(t, coefs[3, :]), #diffusion_coefficient,
        projection=projection,
        len_history=config["model_args"]["len_history"],
        dev=dev
    );

    ps, st = Lux.setup(rng, model) .|> dev;

    weights_and_states = checkpoint_manager.load_model();
    ps = weights_and_states.ps |> dev;
    st = weights_and_states.st |> dev;

    ##### Test stochastic interpolant #####
    st_ = Lux.testmode(st);
    num_test_paths = 4;

    print("Number of generator steps: ", num_generator_steps)

    gif_save_path = "output_$diffusion_multiplier";

    print("\n")
    @time compare_sde_pred_with_true(
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
    print("####################################################")
    print("\n")

    CUDA.reclaim()
    GC.gc()

end;

CUDA.reclaim()
GC.gc()
