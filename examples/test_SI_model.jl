# ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "20GiB"
# ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"

using StochasticInterpolants
using Lux
using YAML
using Random
using NPZ
using LuxCUDA
using Optimisers
using FileIO
using BenchmarkTools

CUDA.reclaim()
GC.gc()

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();
cpu_dev = LuxCPUDevice();


# Choose between "transonic_cylinder_flow", "incompressible_flow", "turbulence_in_periodic_box"
test_case = "kolmogorov";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
mask = mask |> dev;

num_train = size(trainset, 5);
num_steps = size(trainset, 4);
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3);


##### Hyperparameters #####
continue_training = false;
model_base_dir = "trained_models/";
model_name = "forecasting_model_follmer_interpolant";
model_name = "forecasting_model_optimal_interpolant";

checkpoint_manager = CheckpointManager(
    test_case, model_name; base_folder=model_base_dir
);

config = checkpoint_manager.neural_network_config

##### Forecasting SI model #####
# Define the velocity model
velocity = get_SI_neural_network(;
    image_size=(H, W),
    model_params=config["model_args"]
);

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
        -1.49748  -0.181875   0.127546  -0.0188962  -0.0104655  -0.00495827  -0.00481044  -0.00316509;
        -1.89775   0.182452  -0.224247   0.0196697  -0.0101798   0.00545455  -0.00265759   0.00324879;
        -4.94467  -1.35222   -0.68816   -0.400287   -0.276452   -0.174084    -0.11204     -0.0665296
    ];
    # Cast to float32
    coefs = coefs .|> T;

    interpolant = Interpolant(
        t -> get_alpha_series(t, coefs[1, :]), 
        t -> get_beta_series(t, coefs[2, :]), 
        t -> get_dalpha_series_dt(t, coefs[1, :]),
        t -> get_dbeta_series_dt(t, coefs[2, :]), 
        t -> get_gamma_series(t, coefs[3, :]),
        t -> get_dgamma_series_dt(t, coefs[3, :])
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

using Plots

# Define interpolant and diffusion coefficients
diffusion_multiplier = 1.1f0;
interpolant_multiplier = 1.1f0;


gamma = t -> interpolant_multiplier.* (1f0 .- t);
dgamma_dt = t -> -1f0 .* interpolant_multiplier; #ones(size(t)) .* diffusion_multiplier;
diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));


# gamma = t -> interpolant_multiplier .* sqrt.(2f0 .*t .* (1f0 .- t));
# dgamma_dt = t -> -1f0 .* interpolant_multiplier; #ones(size(t)) .* diffusion_multiplier;
# diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));
num_steps = 200;

alpha = t -> 1f0 .- t; 
dalpha_dt = t -> -1f0;

beta = t -> t.^2;
dbeta_dt = t -> 2f0 .* t;



# alpha = t -> cos.(0.5f0 .* pi .* t); 
# dalpha_dt = t -> -0.5f0 .* pi .* sin.(0.5f0 .* pi .* t);

# beta = t -> sin.(0.5f0 .* pi .* t)
# dbeta_dt = t -> 0.5f0 .* pi .* cos.(0.5f0 .* pi .* t)

interpolant = Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt);

t_vec = range(0f0, 1f0, length=num_steps);
# plot(sqrt.(t_vec) .* gamma(t_vec))

t_vec = reshape(t_vec, 1, 1, 1, num_steps);

x0_ = testset[:, :, :, 1, :];
x1 = testset[:, :, :, 2, :];

x0 = repeat(x0_, outer = [1, 1, 1, num_steps]);
x1 = repeat(x1, outer = [1, 1, 1, num_steps]);

noise = randn(size(x0)[1:3]..., 1);
noise = repeat(noise, outer = [1, 1, 1, num_steps]);

norm(x) = sqrt(sum(x.^2));

test_I = interpolant.interpolant(x0, x1, t_vec) + sqrt.(t_vec) .* noise .* interpolant.gamma(t_vec);
test_I = reshape(test_I, size(test_I)[1:4]..., 1);
x = sqrt.(test_I[:, :, 1, :, 1].^2 + test_I[:, :, 2, :, 1].^2);
# preds_to_save = (x, );
# create_gif(
#     preds_to_save, 
#     "interpolant" * ".gif",
#     ["True", ]
# )
using Plots
energy_true = compute_total_energy(test_I);
true_energy = []
pred = []
for i = 1:1
    noise = randn(size(x0)[1:3]..., 1);
    noise = repeat(noise, outer = [1, 1, 1, num_steps]);

    test_I = interpolant.interpolant(x0, x1, t_vec) + sqrt.(t_vec) .* noise .* interpolant.gamma(t_vec);
    test_I = reshape(test_I, size(test_I)[1:4]..., 1);

    energy_true = compute_total_energy(test_I);
    push!(true_energy, energy_true)

    e = energy_true[1, 1]
    push!(pred, e)
    dt = 1f0 / num_steps
    for t_i = 1:num_steps-1

        tt = t_vec[1, 1, 1, t_i]

        x_0_energy = sum(x0[:, :, 1, 1].^2) + sum(x0[:, :, 2, 1].^2)
        # x_0_energy /= 2
        rhs = dalpha_dt(tt) .* alpha(tt) .* x_0_energy
        

        x_1_energy = sum(x1[:, :, 1, 1].^2) + sum(x1[:, :, 2, 1].^2)
        # x_1_energy /= 2
        rhs += dbeta_dt(tt) .* beta(tt) .* x_1_energy

        x0_x1_prod = sum(x0[:, :, 1, 1] .* x1[:, :, 1, 1]) + sum(x0[:, :, 2, 1] .* x1[:, :, 2, 1])
        
        rhs += (dbeta_dt(tt) * alpha(tt) + dalpha_dt(tt) * beta(tt)) * x0_x1_prod

        rhs += 0.5 * 128*128*2*gamma(tt).^2

        rhs += dgamma_dt(tt) * gamma(tt) * tt * 128*128*2

        lambda_term = x_0_energy - 0.5 * norm(test_I[:, :, :, t_i, 1])^2

        e = e + rhs * dt + dt * gamma(tt) * lambda_term

        push!(pred, e)
    end
    
    # energy[j, i] = sum(sol[:, :, 1, j, i].^2) + sum(sol[:, :, 2, j, i].^2)
    # energy[j, i] /= 2

end
plot(true_energy, label="")
plot!(pred, label="")



plot_list = [];
p1 = heatmap(
    x[:, :, 1], legend=false, xticks=false, yticks=false, 
    clim=(minimum(x), maximum(x)),
    aspect_ratio=:equal, 
    colorbar=true, color=cgrad(:Spectral_11, rev=true)
)
p2 = heatmap(
    x[:, :, end], legend=false, xticks=false, yticks=false, 
    clim=(minimum(x), maximum(x)),
    aspect_ratio=:equal, 
    colorbar=true, color=cgrad(:Spectral_11, rev=true)
)
p3 = heatmap(
    x[:, :, 1]-x[:, :, end], legend=false, xticks=false, yticks=false, 
    aspect_ratio=:equal, 
    colorbar=true, color=cgrad(:Spectral_11, rev=true)
)

p = plot(p1, p2, p3,layout=(1, 3), size=(1600, 800))


x0_energy = sum(x0[:, :, 1, 1].^2) + sum(x0[:, :, 2, 1].^2);
x0_energy /= 2

x1_energy = sum(x1[:, :, 1, 1].^2) + sum(x1[:, :, 2, 1].^2);
x1_energy /= 2


