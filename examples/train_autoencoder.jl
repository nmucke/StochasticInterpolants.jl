ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "20GiB"
ENV["TMPDIR"] = "/export/scratch1/ntm/postdoc/StochasticInterpolants.jl/tmp"

using StochasticInterpolants
using Lux
using Random
using LuxCUDA
using Optimisers
using FileIO
using Statistics
using Zygote
using CUDA
using Plots
using YAML
using OrderedCollections

# For running on CPU.
# Consider reducing the sizes of DNS, LES, and CNN layers if
# you want to test run on a laptop.
T = Float32;
ArrayType = Array;
device = identity;
clean() = nothing;

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

# Number of training samples
num_train = size(trainset, 5);
num_steps = size(trainset, 4);
H, W, C = size(trainset, 1), size(trainset, 2), size(trainset, 3);

trainset = reshape(trainset, H, W, C, num_train*num_steps);

num_test = size(testset, 5);
num_test_steps = size(testset, 4);
testset = testset[:, :, :, :, 1];

# testset = reshape(testset, H, W, C, num_test*num_test_steps);
# test_ids = shuffle(rng, 1:size(testset)[end]);
# testset = testset[:, :, :, test_ids];

##### Hyperparameters #####
continue_training = false;
model_base_dir = "trained_models/";
model_name = "VAE";

if continue_training
    checkpoint_manager = CheckpointManager(
        test_case, model_name; base_folder=model_base_dir
    )

    config = checkpoint_manager.neural_network_config("trained_models/$test_case/$model_name")
else
    config = YAML.load_file("configs/variational_autoencoders/$test_case.yml")
    
    checkpoint_manager = CheckpointManager(
        test_case, model_name; 
        neural_network_config=config, 
        data_config=YAML.load_file("configs/test_cases/$test_case.yml"),
        base_folder=model_base_dir
    )
end;

##### Autoencoder #####
autoencoder = VariationalAutoencoder(
    in_channels=config["model_args"]["in_channels"],
    image_size=(H, W),
    num_latent_channels=config["model_args"]["num_latent_channels"],
    channels=config["model_args"]["channels"], 
    padding=config["model_args"]["padding"],
);

##### Optimizer #####
opt = Optimisers.AdamW(
    T(config["optimizer_args"]["learning_rate"]), 
    (0.9f0, 0.99f0), 
    T(config["optimizer_args"]["weight_decay"])
);

##### Load model #####
continue_training = false;
if continue_training
    weights_and_states = checkpoint_manager.load_model();
    ps, st, opt_state = weights_and_states .|> dev;
else
    ps, st = Lux.setup(rng, autoencoder) .|> dev;
    opt_state = Optimisers.setup(opt, ps);
end;

##### Loss #####
loss_fn_VAE(x, model, ps, st) = begin
    (latent_mean, latent_log_var), st = model.encode(x, ps, st)

    beta = 1e-2
    alpha = 1e-2

    # latent_std = exp.(0.5 .* latent_std)

    z = randn(rng, size(latent_mean)) |> dev
    z = latent_mean + latent_log_var .* z

    x_recon, st = model.decode(z, ps, st)

    reconstruction_loss = mean((x - x_recon).^2)

    KL_divergence = -0.5f0 .* sum(1 .+  latent_log_var .- latent_mean.^2 .- exp.(latent_log_var), dims=(1,2,3))
    KL_divergence = beta .* mean(KL_divergence)

    (latent_pred, _), st = model.encode(x_recon, ps, st)
    consistency = sum((latent_mean - latent_pred).^2, dims=(1,2,3))
    consistency = alpha .* mean(consistency)
        
    loss = reconstruction_loss + KL_divergence + consistency
    return loss, st
end;


##### Training #####
best_loss = 1e8;
batch_size = config["training_args"]["batch_size"];
recon_error_dict = OrderedDict();
loss_vec = [];
for epoch in 1:config["training_args"]["num_epochs"]
    running_loss = 0.0

    # Shuffle trainset
    train_ids = shuffle(rng, 1:size(trainset)[end])
    trainset = trainset[:, :, :, train_ids]

    for i in 1:batch_size:size(trainset)[end]
        if i + batch_size - 1 > size(trainset)[end]
            break
        end

        x = trainset[:, :, :, i:i+batch_size-1] |> dev

        (loss, st), pb_f = Zygote.pullback(
            p -> loss_fn_VAE(x, autoencoder, p, st), ps
        );

        running_loss += loss

        gs = pb_f((one(loss), nothing))[1];
        opt_state, ps = Optimisers.update!(opt_state, ps, gs)
        
    end

    running_loss /= floor(Int, size(trainset)[end] / batch_size)

    loss_vec = vcat(loss_vec, running_loss)
    fig = plot(loss_vec, yaxis=:log)
    checkpoint_manager.save_figure(fig, "training_loss.png")

    if epoch % 1 == 0
        print("Loss Value after $epoch iterations: $running_loss \n")
    end

    if epoch % config["training_args"]["test_frequency"] == 0
        # x = testset[:, :, :, 1:1+batch_size-1] |> dev

        CUDA.reclaim()
        GC.gc()

        testset = testset |> dev;

        st_ = Lux.testmode(st)
        (latent_mean, latent_log_var), st = autoencoder.encode(testset, ps, st_)

        x_recon, st = autoencoder.decode(latent_mean, ps, st_)

        create_gif(
            (latent_mean[:, :, 1, :] |> cpu_dev, x_recon[:, :, 1, :] |> cpu_dev, testset[:, :, 1, :] |> cpu_dev), 
            "$(checkpoint_manager.figures_dir)/$(epoch)_anim.mp4", 
            ("latent", "recon", "true")
        )        

        recon_error = mean((testset - x_recon).^2)
        print("Reconstruction error: $recon_error \n")
        
        x_recon = x_recon |> cpu_dev;
        testset = testset |> cpu_dev;
        x_recon_plot = sqrt.(x_recon[:, :, 1, 1].^2 .+ x_recon[:, :, 2, 1].^2)
        x_true_plot = sqrt.(testset[:, :, 1, 1].^2 .+ testset[:, :, 2, 1].^2)

        p1 = heatmap(x_recon_plot)
        p2 = heatmap(x_true_plot)
        p_list = [p1, p2]
        p = plot(p_list..., layout=(1, 2), size=(800, 400))
        savefig("$(checkpoint_manager.figures_dir)/$(epoch)_recon.png")

        energy_true = compute_total_energy(testset) |> cpu_dev;
        energy_pred = compute_total_energy(x_recon) |> cpu_dev;
        plot(energy_true, color=:blue, label="True", linewidth=3)
        plot!(energy_pred, color=:red, label="Pred", linewidth=3)
        savefig("$(checkpoint_manager.figures_dir)/$(epoch)_energy.png")

        energy_pred_latent = compute_total_energy(latent_mean) |> cpu_dev;
        plot(energy_pred_latent, color=:blue, label="True", linewidth=3)
        savefig("$(checkpoint_manager.figures_dir)/$(epoch)_latent_energy.png")

        # Save model if it is the best so far
        if recon_error < best_loss
            checkpoint_manager.save_model(ps, st, opt_state)
            info_dict = OrderedDict(
                "Model saved at epoch" => epoch,
                "test_pathwise_MSE" => recon_error,
            )
            checkpoint_manager.write_dict_to_txt_file(info_dict, "best_model_info.txt")

            best_loss = recon_error
            early_stop_counter = 0
        end
    end

    CUDA.reclaim()
    GC.gc()
end


















testset = testset |> dev;

st_ = Lux.testmode(st)
(latent_mean, latent_log_var), st = autoencoder.encode(testset, ps, st_)




x = trainset[:, :, :, 1:1+batch_size-1] |> dev;

(x_mean, x_std), st = autoencoder.encode(x, ps, st)

BCE_loss(x, x)

x_dec, st = autoencoder(x, ps, st_);


x = x |> cpu_dev;
x_dec = x_dec |> cpu_dev;
heatmap(x[:, :, 1, 1])
heatmap(x_dec[:, :, 1, 1])