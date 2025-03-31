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

using ForwardDiff

using Statistics
using Plots

test_case = "kolmogorov";
# test_case = "incompressible_flow";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "default";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);
# mask = mask |> dev;
num_train = size(trainset, 5);
num_steps = size(trainset, 4);
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3);


CUDA.reclaim()
GC.gc()

# Set the seed
rng = Random.default_rng();
Random.seed!(rng, 0);

# Get the device determined by Lux
dev = gpu_device();

cpu_dev = LuxCPUDevice();

##### Hyperparameters #####
continue_training = false;
model_base_dir = "trained_models/";
model_name = "forecasting_model_new";

if continue_training
    checkpoint_manager = CheckpointManager(
        test_case, model_name; base_folder=model_base_dir
    )

    config = checkpoint_manager.neural_network_config("trained_models/$test_case/$model_name")
else
    config = YAML.load_file("configs/neural_networks/$test_case.yml")
    
    checkpoint_manager = CheckpointManager(
        test_case, model_name; 
        neural_network_config=config, 
        data_config=YAML.load_file("configs/test_cases/$test_case.yml"),
        base_folder=model_base_dir
    )
end;
CUDA.allowscalar()

trainset = prepare_data_for_time_stepping(
    trainset,
    trainset_pars;
    len_history=config["model_args"]["len_history"]
);

x_0 = trainset.init_distribution[:, :, :, end, :] .|> Float32;
x_1 = trainset.target_distribution .|> Float32;

num_samples = size(x_0)[end]

interpolant(coefs, dev=gpu_device()) = begin

    num_total_coefs = length(coefs)
    coefs = reshape(coefs, 2, num_total_coefs ÷ 2)

    alpha = t -> get_alpha_series(t, coefs[1, :] |> dev)
    beta = t -> get_beta_series(t, coefs[2, :] |> dev)
    # gamma = t -> get_gamma_series(t, coefs[3, :] |> dev)

    dalpha_dt = t -> get_dalpha_series_dt(t, coefs[1, :] |> dev)
    dbeta_dt = t -> get_dbeta_series_dt(t, coefs[2, :] |> dev)
    # dgamma_dt = t -> get_dgamma_series_dt(t, coefs[3, :] |> dev)

    gamma = t -> 0.1 .*(1f0 .- t)
    dgamma_dt = t -> -0.1 .* 1f0

    return Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt) 
end

lambda = 1f-8
mu = 1f-12
mu_max = 1e4
beta = 2

num_spochs = 5
batch_size = 256

# i = 1
# x_0_batch = x_0[:, :, :, i:i+batch_size-1];
# x_1_batch = x_1[:, :, :, i:i+batch_size-1];

# t = rand!(rng, similar(x_1, 1, 1, 1, batch_size));

# I_velocity = x -> mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x)).^2)
# lol = I_velocity(coefs)

# interpolant_energy_evolution = coefs -> d_interpolant_energy_dt(
#     x_0_batch,
#     x_1_batch,
#     t,
#     interpolant(coefs)
# )
# lal = interpolant_energy_evolution(coefs)

# grad = ForwardDiff.gradient(I_velocity, coefs)



# i=10
# t = rand!(rng, similar(x_1, 1, 1, 1, batch_size)) |> dev;
# x_0_batch = x_0[:, :, :, i:i+batch_size-1] |> dev;
# x_1_batch = x_1[:, :, :, i:i+batch_size-1] |> dev;
# I_velocity = x -> mean(d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(x[:, :, 1:2, :]), omega).^2)
# e = I_velocity(coefs)

# I_velocity = x -> mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x)).^2)
# v = I_velocity(coefs)

# e/v/1e3

# grad = ForwardDiff.gradient(I_velocity, coefs)
# hessian = ForwardDiff.hessian(I_velocity, coefs)
# hessian \ grad

# coefs -= hessian \ grad

if test_case == "kolmogorov"
    omega = [0.04908f0, 0.04908f0]
elseif test_case == "incompressible_flow"
    omega = [0.03125f0, 0.03125f0]
end

num_coefs = 5
best_coefs = randn(num_coefs * 2);
best_coefs = best_coefs .|> Float32;
best_coefs = best_coefs |> dev;

best_energy = 1e8

for num_coefs in [5, ]
    for iter in [1, 2, 3, 4, 5]

        lambda = 6.5f0

        coefs = randn(num_coefs * 2);
        coefs = coefs .|> Float32;
        coefs = coefs |> dev;

        for epoch = 1:num_spochs

            shuffled_ids = shuffle(rng, 1:size(x_0)[end])
            x_0 = x_0[:, :, :, shuffled_ids]
            x_1 = x_1[:, :, :, shuffled_ids]

            running_loss = 0f0
            for i in 1:batch_size:size(x_0)[end]
                if i + batch_size - 1 > size(x_0)[end]
                    break
                end

                x_0_batch = x_0[:, :, :, i:i+batch_size-1] |> dev;
                x_1_batch = x_1[:, :, :, i:i+batch_size-1] |> dev;
                
                t = rand!(rng, similar(x_0_batch, 1, 1, 1, batch_size)) |> dev;

                # I_velocity = x -> mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x)).^2)  
                obj = x -> mean(d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(x), omega).^2) + mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x), omega).^2) + lambda * sum(x.^2)
                # obj = x -> mean(d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(x), omega).^2) + lambda * sum(x.^2)
                # obj = x -> mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x), omega).^2) + lambda * sum(x.^2)
                # obj = x -> mean(d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(x), omega).^2) + mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x), omega).^2)# + lambda * sum(x.^2)
                
                # obj = x -> mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x), omega).^2) + 10f0 * sum(x.^2)
                # obj = x -> mean(d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(x)).^2) + 10f0 * sum(x.^2)
                # grad = ForwardDiff.gradient(I_velocity, coefs)
                # coefs -= lr .* grad

                grad = ForwardDiff.gradient(obj, coefs)
                hessian = ForwardDiff.hessian(obj, coefs)

                hess_inv_grad = hessian \ grad

                # Line search for step size
                step_size = 1f0
                coefs = coefs - step_size .* hess_inv_grad
                new_coefs = coefs
                old_loss = obj(coefs)
                for j in 1:15
                    new_coefs = coefs - step_size .* hess_inv_grad
                    new_loss = obj(new_coefs)
                    if new_loss < old_loss
                        # print("j: $j, Loss: $new_loss, Step Size: $step_size, old loss: $old_loss")
                        # print("\n")
                        break
                    end
                    step_size /= 2
                end

                coefs = new_coefs

            end

            # println("Epoch: $epoch, Loss: $(running_loss / batch_size)")
        end
        total_energy = 0f0
        total_velocity = 0f0

        t = rand!(rng, similar(x_0, 1, 1, 1, batch_size)) |> dev;

        counter = 0
        for i in 1:batch_size:size(x_0)[end]
            if i + batch_size - 1 > size(x_0)[end]
                break
            end

            x_0_batch = x_0[:, :, :, i:i+batch_size-1] |> dev;
            x_1_batch = x_1[:, :, :, i:i+batch_size-1] |> dev;

            energ = x -> mean(d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(x), omega).^2)
            velocity = x -> mean(interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(x), omega).^2)

            total_energy += energ(coefs)
            total_velocity += velocity(coefs)

            counter += 1
        end

        if total_energy < best_energy
            best_energy = total_energy
            best_coefs = coefs
        end

        println("Num coefs: $num_coefs, Lambda: $lambda, Energy: $(total_energy / counter), Velocity: $(total_velocity / counter)")
    end
end



# -1.49748  -0.181875   0.127546  -0.0188962  -0.0104655  -0.00495827  -0.00481044  -0.00316509
# -1.89775   0.182452  -0.224247   0.0196697  -0.0101798   0.00545455  -0.00265759   0.00324879
# -4.94467  -1.35222   -0.68816   -0.400287   -0.276452   -0.174084    -0.11204     -0.0665296

coefs = reshape(best_coefs, 2, 5)

coefs_energy = [
    # 18.1921   23.337    7.50453  -2.39351  -1.5831
    # -25.392   -24.2777  -6.41302   2.82029   1.46301
    # -1.05701  -0.00180631   -0.0302278  -0.0021648   -0.00841667
    # -1.05753   0.000520512  -0.0297646   0.00132788  -0.0046216
    # -0.947154  -5.63862  -0.869923  -2.48371   0.228655
    # -1.17808    5.63254   0.800721   2.48293  -0.242225
    -1.05987  -0.000857944  -0.0283922  -0.000478533  -0.00400515
    -1.06167   6.36467e-5   -0.0341536   0.000724897  -0.0100014
] .|> Float32


coefs_transport = [
    -0.0586196  0.00959212  -0.00419747  0.00131925  0.00117193
    -0.058586   0.0096501   -0.00416698  0.00145934  0.00101391
] .|> Float32

# coefs = [
#     -1.22202  -0.0658369  -0.0373067  -0.0114895   -0.0213044;
#     -1.22062  -0.0615028  -0.0364064  -0.00311659   0.00803215
# ]


coefs_energy_transport = [
    -1.05734  -0.00348673  -0.0312818  -0.00382112  -0.00580364;
    -1.05611   0.00127347  -0.0293777   0.00343358  -0.00645624
] .|> Float32
# coefs = [
#     -0.991808  -0.00170546  -0.0325862  -0.00137419   -0.00699029
#     -0.991881   0.00141638  -0.0326092  -0.000478376  -0.00664719
# ]




t_vec = LinRange(0f0, 1f0, num_steps)
t_all = reshape(t_vec, 1, 1, 1, num_steps)


coefs = coefs |> cpu_dev;

x0 = testset[:, :, :, 10:10, 1]
x1 = testset[:, :, :, 11:11, 1]

x0_energy = 0.5*sum(x0.^2, dims=(1, 2, 3)) * omega[1] * omega[2]

num_steps = 1000
function compute_energy_evolution(coefs, x0_energy=x0_energy)
    num_steps = 1000
    t_vec = LinRange(0f0, 1f0, num_steps)
    dt = 1f0 / num_steps

    alpha = t -> get_alpha_series(t, coefs[1, :])
    beta = t -> get_beta_series(t, coefs[2, :])
    # plot(t_vec, alpha.(t_vec), label="Alpha", xlabel="Time", ylabel="Alpha", title="Alpha")
    # plot!(t_vec, beta.(t_vec), label="Beta", xlabel="Time", ylabel="Beta", title="Beta")

    energy_pred_list = zeros(num_steps)
    energy_true = []
    for (i, t) in enumerate(t_vec)
        energy_t = x0_energy .+ dt .* d_interpolant_energy_dt(
            x0, x1, t, interpolant(coefs, cpu_dev), omega, cpu_dev
        )
        energy_pred_list[i] = energy_t[1, 1, 1, 1]
        x0_energy = energy_t
    end

    return energy_pred_list
end

function compute_energy_evolution_true(coefs, x0, x1)
    noise = randn(size(x0)[1:3]..., 1);
    test_I = interpolant(coefs).interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* interpolant(coefs).gamma(t_all);
    test_I = reshape(test_I, size(test_I)[1:4]..., 1);
    
    energy_true = compute_total_energy(test_I, omega);

    return energy_true
end

using Plots
using LaTeXStrings


x0 = testset[:, :, :, 10:10, 1]
x1 = testset[:, :, :, 11:11, 1]

num_cases = 100

energy_transport = zeros(num_steps, Int(num_cases/10)*5)
energy_energy = zeros(num_steps, Int(num_cases/10)*5)
energy_energy_transport = zeros(num_steps, Int(num_cases/10)*5)
energy_true_non_optimized = zeros(num_steps, Int(num_cases/10)*5)
energy_true_non_optimized_linear = zeros(num_steps, Int(num_cases/10)*5)

inter = get_interpolant(
    config["interpolant_args"]["alpha"],
    config["interpolant_args"]["beta"],
    config["interpolant_args"]["gamma"],
    config["interpolant_args"]["gamma_multiplier"],
);

# test_I = inter.interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* inter.gamma(t_all);
# test_I = reshape(test_I, size(test_I)[1:4]..., 1);

# energy_true_non_optimized = compute_total_energy(test_I, omega);


# noise = randn(size(x0)[1:3]..., 1);
# test_I = interpolant(coefs_transport).interpolant(x0, x1, t_all) .+ sqrt.(t_all) .* noise .* interpolant(coefs_transport).gamma(t_all);
# test_I = reshape(test_I, size(test_I)[1:4]..., 1);

# energy_true = compute_total_energy(test_I, omega);


# noise = randn(size(x0)[1:3]..., 1);
# test_I = inter.interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* inter.gamma(t_all);
# test_I = reshape(test_I, size(test_I)[1:4]..., 1);

# energy_true_non_optimized[:, 1] = compute_total_energy(test_I, omega);

counter = 1
for i in 1:10:num_cases
    for j in 1:5
        x0 = testset[:, :, :, i:i, j]
        x1 = testset[:, :, :, i+1:i+1, j]

        # e = compute_energy_evolution_true(coefs_transport, x0, x1) |> cpu
        energy_transport[:, counter] = compute_energy_evolution_true(coefs_transport, x0, x1) |> cpu
        energy_energy[:, counter] = compute_energy_evolution_true(coefs_energy, x0, x1)
        energy_energy_transport[:, counter] = compute_energy_evolution_true(coefs_energy_transport, x0, x1)

        noise = randn(size(x0)[1:3]..., 1);
        test_I = inter.interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* inter.gamma(t_all);
        test_I = reshape(test_I, size(test_I)[1:4]..., 1);
        
        energy_true_non_optimized[:, counter] = compute_total_energy(test_I, omega);

        inter_linear = get_interpolant(
            config["interpolant_args"]["alpha"],
            "linear",
            config["interpolant_args"]["gamma"],
            config["interpolant_args"]["gamma_multiplier"],
        );
        test_I = inter_linear.interpolant(x0, x1, t_all) + sqrt.(t_all) .* noise .* inter.gamma(t_all);
        test_I = reshape(test_I, size(test_I)[1:4]..., 1);
        
        energy_true_non_optimized_linear[:, counter] = compute_total_energy(test_I, omega);

        
        counter += 1
    end
end


# velocity_transport = zeros(num_steps, num_cases*5)
# velocity_energy = zeros(num_steps, num_cases*5)
# velocity_energy_transport = zeros(num_steps, num_cases*5)
# velocity = zeros(num_steps, num_cases*5)

# counter = 1
# for i in 1:10:num_cases
#     for j in 1:5
#         x0 = testset[:, :, :, i:i, j]
#         x1 = testset[:, :, :, i+1:i+1, j]

#         t = rand!(rng, similar(x_0, 1, 1, 1, batch_size)) |> dev;
#         velocity[:, counter] = [mean(interpolant_velocity(x0 |> dev, x1 |> dev, t * ones(1,1,1,1) |> dev, inter, omega).^2) for t in t_vec]
#         velocity_transport[:, counter] = [mean(interpolant_velocity(x0 |> dev, x1 |> dev, t * ones(1,1,1,1) |> dev, interpolant(coefs_transport), omega).^2) for t in t_vec]
#         velocity_energy[:, counter] = [mean(interpolant_velocity(x0 |> dev, x1 |> dev, t * ones(1,1,1,1) |> dev, interpolant(coefs_energy), omega).^2) for t in t_vec]
#         velocity_energy_transport[:, counter] = [mean(interpolant_velocity(x0 |> dev, x1 |> dev, t * ones(1,1,1,1) |> dev, interpolant(coefs_energy_transport), omega).^2) for t in t_vec]

#         counter += 1
#     end
# end


noise = randn(size(x0)[1:3]..., 1);
noise = repeat(noise, outer = [1, 1, 1, num_steps]);


Plots.plot(t_vec, mean(energy_true_non_optimized, dims=2), label="Not optimized (quadratic)", xlabel="Pseudo-time", ylabel="Energy", labelsize=2, linewidth=3, ylims=(20, 45), legendfontsize=10)
# Plots.plot!(t_vec, mean(energy_transport, dims=2), label="Tranport Optimized", linewidth=3)
# Plots.plot!(t_vec, mean(energy_energy, dims=2), label="Energy Optimized", linewidth=3)
# Plots.plot!(t_vec, mean(energy_true_non_optimized_linear, dims=2), label="Not optimized (linear)", linewidth=3, legendfontsize=10)
Plots.plot!(t_vec, mean(energy_energy_transport, dims=2), label="Optimized", linewidth=3, legendfontsize=10)
Plots.savefig("interpolant_figures/energy_evolution.pdf")

# println("Velocity: ", mean(velocity))
# println("Velocity Transport: ", mean(velocity_transport))
# println("Velocity Energy: ", mean(velocity_energy))
# println("Velocity Energy Transport: ", mean(velocity_energy_transport))


# Plots.plot(t_vec, mean(velocity), label="Not optimized", xlabel="Pseudo-time", ylabel="Transport cost", linewidth=3)
# Plots.plot!(t_vec, mean(velocity_transport), label="Transport optimized", linewidth=3)
# Plots.plot!(t_vec, mean(velocity_energy), label="Energy optimized", linewidth=3)
# Plots.plot!(t_vec, mean(velocity_energy_transport), label="Energy and Transport optimized",  linewidth=3)


interpolant_transport = interpolant(coefs_transport)
interpolant_energy = interpolant(coefs_energy)
interpolant_energy_transport = interpolant(coefs_energy_transport)

inter_linear = get_interpolant(
    config["interpolant_args"]["alpha"],
    "linear",
    config["interpolant_args"]["gamma"],
    config["interpolant_args"]["gamma_multiplier"],
);

Plots.plot(t_vec, inter.alpha(t_vec), label="Not optimized", xlabel="Pseudo-time", ylabel=L"\alpha_{\tau}", linewidth=3)
# Plots.plot!(t_vec, interpolant_transport.alpha(t_vec), label="Transport optimized",  linewidth=3)
# Plots.plot!(t_vec, interpolant_energy.alpha(t_vec), label="Energy optimized", linewidth=3)
Plots.plot!(t_vec, interpolant_energy_transport.alpha(t_vec), label="Optimized",linewidth=3)
Plots.savefig("interpolant_figures/alpha.pdf")


Plots.plot(t_vec, inter.beta(t_vec), label="Not optimized (quadratic)", xlabel="Pseudo-time", ylabel=L"\beta_{\tau}", linewidth=3)
# Plots.plot!(t_vec, interpolant_transport.beta(t_vec), label="Transport optimized", linewidth=3)
# Plots.plot!(t_vec, interpolant_energy.beta(t_vec), label="Energy optimzied", linewidth=3)

# Plots.plot!(t_vec, inter_linear.beta(t_vec), label="Not optimized (linear)", linewidth=3)
Plots.plot!(t_vec, interpolant_energy_transport.beta(t_vec), label="Optimized", linewidth=3)
Plots.savefig("interpolant_figures/beta.pdf")




min_velocity = minimum(sqrt.(testset[:, :, 1, :, :].^2 + testset[:, :, 2, :, :].^2))
max_velocity = maximum(sqrt.(testset[:, :, 1, :, :].^2 + testset[:, :, 2, :, :].^2))


min_x = minimum(testset[:, :, :, :, :])
max_x = maximum(testset[:, :, :, :, :])

step = 0.1
t = step * ones(1,1,1,1) |> dev
velocity_magnitude = interpolant(coefs_energy_transport).interpolant(x0 |> dev, x1 |> dev, t)

using CairoMakie
using GLMakie
times_to_plot = [0, 0.2, 0.4, 0.6, 0.8, 1]
n_plots = 6

fig_width = 400 * n_plots # 200 pixels per plot
fig_height = 400 # Keep height constant
fig = CairoMakie.Figure(; size = (fig_width, fig_height))
hm = nothing
for (i, time_step) in enumerate(times_to_plot)
    if i == 1
        y_label = "Optimized interpolant"
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
    t = time_step * ones(1,1,1,1) |> dev
    inter = interpolant(coefs_energy_transport)

    # inter = get_interpolant(
    #     config["interpolant_args"]["alpha"],
    #     config["interpolant_args"]["beta"],
    #     config["interpolant_args"]["gamma"],
    #     config["interpolant_args"]["gamma_multiplier"],
    # );

    velocity_magnitude = inter.dinterpolant_dt(x0 |> dev, x1 |> dev, t)

    println(t)

    z = randn(size(x0)) |> dev
    noise = sqrt.(t) .* z .* inter.gamma(t)
    velocity_magnitude = velocity_magnitude + noise
    # velocity_magnitude = sqrt.(velocity_magnitude[:, :, 1, 1].^2 + velocity_magnitude[:, :, 2, 1].^2) |> cpu
    velocity_magnitude = velocity_magnitude[:, :, 1, 1] |> cpu
    hm = GLMakie.heatmap!(
        ax, 
        velocity_magnitude; 
        colormap = Reverse(:Spectral_11), 
        # colorrange = (min_velocity, max_velocity),
        colorrange = (-0.7, 0.7),
    )
end
Colorbar(fig[1, end+1], hm, ticklabelsize=25, width=30)
CairoMakie.save("interpolant_figures/optimized_interpolant_drift.pdf", fig)











t = 0.5
noise = randn(size(x0)[1:3]..., 1);
noise = repeat(noise, outer = [1, 1, 1, num_steps]);

test_I = interpolant(coefs).interpolant(x0, x1, t) + sqrt.(t) .* noise .* interpolant(coefs).gamma(t);

lagrangian(coefs, t, lambda, mu) = begin
    I_velocity = interpolant_velocity(x_0_batch, x_1_batch, t, interpolant(coefs))
    I_energy = d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(coefs))
    print(maximum(I_velocity))

    return mean(I_velocity, dims=(1, 2, 3)) .+ lambda .* I_energy .+ 0.5f0 .* mu .* I_energy.^2
end

# update coefs
obj = coefs -> lagrangian(coefs, t, lambda, mu)
obj(coefs)


grad = ForwardDiff.gradient(obj, coefs)
coefs -= 0.01 .* grad

# update lambda
lambda = lambda + mu .* d_interpolant_energy_dt(x_0_batch, x_1_batch, t, interpolant(coefs))

# update mu
mu = minimum([beta .* mu, mu_max])
































using Optim

init_coefs = coefs

lol = objective_function(
    reshape(init_coefs, 3, 7),
    x_0,
    x_1
)

lol = objective_function(
    reshape(best_coefs, 3, 7),
    x_0,
    x_1
)

x_0_batch = x_0[:, :, :, 1:4];
x_1_batch = x_1[:, :, :, 1:4];


# algo_st = Newton()
# result = optimize(obj, init_coefs, Optim.Options(show_trace=true, iterations=1000))
# best_coefs = Optim.minimizer(result)

using Optimization, OptimizationMOI #, OptimizationOptimJL
using Ipopt
using JuMP

obj(coefs) = objective_function(reshape(coefs, 3, 7), x_0_batch, x_1_batch)
constrain(coefs) = d_interpolant_energy_dt(reshape(coefs, 3, 7), x_0_batch, x_1_batch)

# optprob = OptimizationFunction(obj, cons = constrain)
# prob = OptimizationProblem(optprob, init_coefs, (x_0_batch, x_1_batch), lcons = [-Inf, ], ucons = [0.0, ])
# sol = solve(prob, IPNewton())

model = Model(Ipopt.Optimizer)
@variable(model, coefs[1:21])
@objective(model, Min, obj(coefs))
@constraint(model, constrain(coefs) == 0)
optimize!(model)
objective_value(model)
best_coefs = value.(coefs)

obj(value.(coefs))
constrain(value.(coefs))







using ForwardDiff
grad = ForwardDiff.gradient(obj, coefs)
hessian = ForwardDiff.hessian(obj, coefs)


using LinearSolve

alpha = 0.01
batch_size = 4
for epoch = range(1,60)
    train_ids = shuffle(rng, 1:size(x_0)[end])

    x_0 = x_0[:, :, :, train_ids];
    x_1 = x_1[:, :, :, train_ids];

    running_loss = 0f0;
    for i in 1:batch_size:size(x_0)[end]
        if i + batch_size - 1 > size(trainset.target_distribution)[end]
            break
        end

        x_0_batch = x_0[:, :, :, i:i+batch_size-1];
        x_1_batch = x_1[:, :, :, i:i+batch_size-1];

        obj = x -> objective_function(reshape(x, 3, 5), x_0_batch, x_1_batch)
        
        grad = ForwardDiff.gradient(obj, coefs)

        if epoch < 500
            coefs -= alpha .* grad
        end
        # else
        #     hessian = ForwardDiff.hessian(obj, coefs)
        #     prob = LinearProblem(hessian, grad)
        #     sol = solve(prob)
        #     coefs = coefs - alpha .* sol
        # end



        running_loss += obj(coefs)
    end

    println("Epoch: $epoch, Loss: $(running_loss / batch_size)")
end




best_coefs = reshape(best_coefs, 3, 7)

coefs = randn(10)
t = LinRange(0f0, 1f0, 100)

lol = get_dbeta_series_dt(t, best_coefs[2, :])
lal = get_beta_series(t, best_coefs[2, :])


using Plots
plot(t, lol, label="dbeta_dt", xlabel="t", ylabel="dbeta_dt", title="dbeta_dt")
plot!(t, lal, label="beta", xlabel="t", ylabel="beta", title="beta")



dalpha_dt = t -> get_dalpha_series_dt(t, coefs)
dbeta_dt = t -> get_dbeta_series_dt(t, coefs)
dgamma_dt = t -> get_dgamma_series_dt(t, coefs)

z = randn!(similar(x_1, size(x_1)))
W = map(t -> sqrt.(t) .* z, t)
velocity = map((t, w) -> dalpha_dt(t) .* x_0 .+ dbeta_dt(t) .* x_1 .+ dgamma_dt(t) .* w, t, W)

using Statistics
velocity_energy = map(i -> mean(velocity[i].^2), range(1, length(t)))
velocity_energy



ForwardDiff.gradient(obj, coefs)

mean_x_0_norm = mean(sum(x_0.^2, dims=(1, 2, 3)))
mean_x_1_norm = mean(sum(x_1.^2, dims=(1, 2, 3)))
mean_inner_product = mean(sum(x_0 .* x_1, dims=(1, 2, 3)))

energy = map(t -> d_interpolant_energy_dt(
    t,
    mean_x_0_norm,
    mean_x_1_norm,
    mean_inner_product,
    interpolant.alpha,
    interpolant.dalpha_dt,
    interpolant.beta,
    interpolant.dbeta_dt,
    interpolant.gamma,
    interpolant.dgamma_dt,
), 
    LinRange(0f0, 1f0, 100)
)


# Get Interpolant
interpolant = get_interpolant(
    config["interpolant_args"]["alpha"],
    config["interpolant_args"]["beta"],
    config["interpolant_args"]["gamma"],
    T(config["interpolant_args"]["gamma_multiplier"]),
);
out = map(t -> d_interpolant_energy_dt(
        t,
        mean_x_0_norm,
        mean_x_1_norm,
        mean_inner_product,
        interpolant.alpha,
        interpolant.dalpha_dt,
        interpolant.beta,
        interpolant.dbeta_dt,
        interpolant.gamma,
        interpolant.dgamma_dt,
    ), 
    LinRange(0f0, 1f0, 100)
)

using Plots
plot(LinRange(0f0, 1f0, 100), out, label="Energy", xlabel="t", ylabel="Energy", title="Energy of the interpolant")

