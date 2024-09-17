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
test_case = "turbulence_in_periodic_box";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

# Load the test case configuration
test_case_config = YAML.load_file("configs/test_cases/$test_case.yml");

# Get data path
data_folder = test_case_config["data_folder"];

# Get dimensions of state and parameter spaces
H = test_case_config["state_dimensions"]["height"];
W = test_case_config["state_dimensions"]["width"];
C = test_case_config["state_dimensions"]["channels"];
if !isnothing(test_case_config["parameter_dimensions"])
    pars_dim = length(test_case_config["parameter_dimensions"]);
    num_pars = pars_dim
else
    pars_dim = 1;
    num_pars = 0;
end;

# Load mask if it exists
if test_case_config["with_mask"]
    mask = npzread("$data_folder/sim_000000/obstacle_mask.npz")["arr_0"];
    mask = permutedims(mask, (2, 1)) |> dev;
else
    mask = ones(H, W, C) |> dev;
end;


# Number of training samples
num_train = length(test_case_config["training_args"]["ids"]);

# Time step information
start_time = test_case_config["training_args"]["time_step_info"]["start_time"];
num_steps = test_case_config["training_args"]["time_step_info"]["num_steps"];
skip_steps = test_case_config["training_args"]["time_step_info"]["skip_steps"];

test_start_time = test_case_config["test_args"][test_args]["time_step_info"]["start_time"];
test_num_steps = test_case_config["test_args"][test_args]["time_step_info"]["num_steps"];
test_skip_steps = test_case_config["test_args"][test_args]["time_step_info"]["skip_steps"];


if test_case == "transonic_cylinder_flow"
    # Load the training data
    trainset, trainset_pars = load_transonic_cylinder_flow_data(
        data_folder=data_folder,
        data_ids=test_case_config["training_args"]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(start_time, num_steps, skip_steps)
    );

    # Load the test data
    testset, testset_pars = load_transonic_cylinder_flow_data(
        data_folder=data_folder,
        data_ids=test_case_config["test_args"][test_args]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(test_start_time, test_num_steps, test_skip_steps)
    );
elseif test_case == "incompressible_flow"
    # Load the training data
    trainset, trainset_pars = load_incompressible_flow_data(
        data_folder=data_folder,
        data_ids=test_case_config["training_args"]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(start_time, num_steps, skip_steps)
    );

    # Load the test data
    testset, testset_pars = load_incompressible_flow_data(
        data_folder=data_folder,
        data_ids=test_case_config["test_args"][test_args]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(test_start_time, test_num_steps, test_skip_steps)
    );

elseif test_case == "turbulence_in_periodic_box"
    # Load the training data
    trainset, trainset_pars = load_turbulence_in_periodic_box_data(
        data_folder=data_folder,
        data_ids=test_case_config["training_args"]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(start_time, num_steps, skip_steps)
    );

    # Load the test data
    testset, testset_pars = load_turbulence_in_periodic_box_data(
        data_folder=data_folder,
        data_ids=test_case_config["test_args"][test_args]["ids"],
        state_dims=(H, W, C),
        num_pars=pars_dim,
        time_step_info=(test_start_time, test_num_steps, test_skip_steps)
    );
else
    error("Invalid test case")
end;

trainset = convert(Array{T}, trainset);
trainset_pars = convert(Array{T}, trainset_pars);
testset = convert(Array{T}, testset);

if test_case_config["normalize_data"]
    # Normalize the data
    normalize_data = StandardizeData(
        test_case_config["norm_mean"], 
        test_case_config["norm_std"],
    );
    trainset = normalize_data.transform(trainset);
    testset = normalize_data.transform(testset);

    # Normalize the parameters
    normalize_pars = NormalizePars(
        test_case_config["pars_min"], 
        test_case_config["pars_max"]
    );
    trainset_pars = normalize_pars.transform(trainset_pars);
    testset_pars = normalize_pars.transform(testset_pars);
else
    normalize_data = nothing;
end;


# # Create a gif
# x1, p1 = trainset[:, :, 1, :, 1], trainset_pars[1, 1, 1];
# x2, p2 = trainset[:, :, 1, :, 2], trainset_pars[1, 1, 2];
# x3, p3 = trainset[:, :, 1, :, 3], trainset_pars[1, 1, 3];
# x4, p4 = trainset[:, :, 1, :, 4], trainset_pars[1, 1, 4];
# x8, p8 = trainset[:, :, 1, :, 5], trainset_pars[1, 1, 5];
# x5, p5 = trainset[:, :, 1, :, 6], trainset_pars[1, 1, 6];
# x6, p6 = trainset[:, :, 1, :, 7], trainset_pars[1, 1, 7];
# x7, p7 = trainset[:, :, 1, :, 8], trainset_pars[1, 1, 8];

# create_gif(
#     (x1, x2, x3, x4, x5, x6, x7, x8), 
#     "HF.gif", 
#     ("Ma = $p1", "Ma = $p2", "Ma = $p3", "Ma = $p4", "Ma = $p5", "Ma = $p6", "Ma = $p7", "Ma = $p8")
# )


len_history = 2;

trainset_init_distribution = zeros(H, W, C, len_history, num_steps-len_history, num_train);
trainset_target_distribution = zeros(H, W, C, num_steps-len_history, num_train);
for i in 1:num_train
    for step = 1:num_steps-len_history
        trainset_init_distribution[:, :, :, :, step, i] = trainset[:, :, :, step:(step+len_history-1), i];
        trainset_target_distribution[:, :, :, step, i] = trainset[:, :, :, step+len_history, i];
    end;
end;
trainset_pars = trainset_pars[:, 1:(num_steps-len_history), :];

trainset_init_distribution = reshape(trainset_init_distribution, H, W, C, len_history, (num_steps-len_history)*num_train);
trainset_target_distribution = reshape(trainset_target_distribution, H, W, C, (num_steps-len_history)*num_train);
trainset_pars_distribution = reshape(trainset_pars, pars_dim, (num_steps-len_history)*num_train);



##### Hyperparameters #####
embedding_dims = 128;
batch_size = 8;
learning_rate = T(1e-4);
weight_decay = T(1e-8);
num_epochs = 2000;
channels = [8, 16, 32, 64];
attention_type = "DiT"; # "linear" or "standard" or "DiT"
use_attention_in_layer = [false, false, false, false]; # [true, true, true, true];
attention_embedding_dims = 64;
num_heads = 4;
projection = project_onto_divergence_free;

##### Forecasting SI model #####
# Define the velocity model
# velocity = DitParsConvNextUNet(
velocity = AttnParsConvNextUNet(
    (H, W); 
    in_channels=C, 
    channels=channels, 
    embedding_dims=embedding_dims, 
    pars_dim=num_pars,
    len_history=len_history,
    attention_type=attention_type,
    use_attention_in_layer=use_attention_in_layer,
    padding="periodic",
    attention_embedding_dims=attention_embedding_dims,
    num_heads=num_heads,
);


# Define interpolant and diffusion coefficients
diffusion_multiplier = 0.25f0;

gamma = t -> diffusion_multiplier.* (1f0 .- t);
dgamma_dt = t -> -1f0 .* diffusion_multiplier; #ones(size(t)) .* diffusion_multiplier;
diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));

alpha = t -> 1f0 .- t; 
dalpha_dt = t -> -1f0;

beta = t -> t.^2;
dbeta_dt = t -> 2f0 .* t;


# Initialise the SI model
model = FollmerStochasticInterpolant(
    velocity; 
    interpolant=Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt),
    diffusion_coefficient=diffusion_coefficient,
    projection=projection,
    dev=dev
);
ps, st = Lux.setup(rng, model) .|> dev;

model_save_dir = "trained_models/forecasting_model";

##### Optimizer #####
opt = Optimisers.AdamW(learning_rate, (0.9f0, 0.99f0), weight_decay);
opt_state = Optimisers.setup(opt, ps);

##### Load checkpoint #####
continue_training = false;
if continue_training
    if test_case == "transonic_cylinder_flow"
        best_model = "checkpoint_transonic_best"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "incompressible_flow"   
        best_model = "best_incompressible_model"
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/$best_model.bson") .|> dev;
    elseif test_case == "turbulence_in_periodic_box"
        continue_epoch = 100;
        ps, st, opt_state = load_checkpoint("trained_models/forecasting_model/checkpoint_epoch_$continue_epoch.bson") .|> dev;
    end;
end;
    
##### Train stochastic interpolant #####
ps, st = train_stochastic_interpolant(
    model=model,
    ps=ps,
    st=st,
    opt_state=opt_state,
    trainset_target_distribution=trainset_target_distribution,
    trainset_init_distribution=trainset_init_distribution,
    trainset_pars_distribution=trainset_pars_distribution,
    testset=testset,
    testset_pars=testset_pars,
    num_test_paths=5,
    model_save_dir=model_save_dir,
    num_epochs=num_epochs,
    batch_size=batch_size,
    normalize_data=normalize_data,
    mask=mask,  
    rng=rng,
    dev=dev
);









possion_matrix_1D = SymTridiagonal(
    -4*ones(size(div, 1)),
    ones(size(div, 1)-1),
);
possion_matrix_1D.ev[1] = 1;
possion_matrix_1D.ev[end, 1] = 1;


using Statistics

dx = 1/(size(trainset[:, :, 1, 1, 1], 1)-1);
dy = 1/(size(trainset[:, :, 1, 1, 1], 2)-1);

# using NonlinearSolve

function divfunc(u, v)
    u_diff_x = (circshift(u, (-1, 0)) - circshift(u, (0, 0))) / dx;
    v_diff_y = (circshift(v, (0, -1)) - circshift(v, (0, 0))) / dy;
    u_diff_x + v_diff_y
end

function gradfunc(q)
    q_grad_x = (circshift(q, (0, 0)) - circshift(q, (1, 0))) / dx;
    q_grad_y = (circshift(q, (0, 0)) - circshift(q, (0, 1))) / dy;
    q_grad_x, q_grad_y
end

function laplacefunc(q)

    # divfunc(gradfunc(q)...)

    # q_laplace = circshift(q, (-2, 0)) + circshift(q, (2, 0)) + circshift(q, (0, -2)) + circshift(q, (0, 2)) - 4q;
    # q_laplace / 4dx^2

    q_laplace = circshift(q, (-1, 0)) + circshift(q, (1, 0)) + circshift(q, (0, -1)) + circshift(q, (0, 1)) - 4q;
    q_laplace / dx^2
end

divfunc(gradfunc(q)...) -laplacefunc(q)

u_int = trainset[:, :, 1, 10, 1:5];
v_int = trainset[:, :, 2, 10, 1:5];
# u_int = (circshift(u_int, (0, 0)) + circshift(u_int, (1, 0))) / 2;
# v_int = (circshift(v_int, (0, 0)) + circshift(v_int, (0, 1))) / 2;


u_int = @. u_int + randn() * 1f-1;
v_int = @. v_int + randn() * 1f-1;

div = divfunc(u_int, v_int);

print("Mean divergence: ", mean(abs.(div)))

q = zeros(size(div));

using LinearAlgebra

div_RHS = reshape(div, size(div)[1]*size(div)[2]);

possion_matrix_1D = Tridiagonal(
    ones(size(div, 1)-1),
    -4*ones(size(div, 1)),
    ones(size(div, 1)-1)
);
possion_matrix_1D = Matrix(possion_matrix_1D);
possion_matrix_1D[1, end] = 1;
possion_matrix_1D[end, 1] = 1;

S = Tridiagonal(
    ones(size(div, 1)-1),
    zeros(size(div, 1)),
    ones(size(div, 1)-1)
);
S = Matrix(S);
S[1, end] = 1;
S[end, 1] = 1;

possion_matrix_2D = kron(I(size(div, 2)), possion_matrix_1D) + kron(S, I(size(div, 2)));
possion_matrix_2D = possion_matrix_2D / (dx * dx);

q = possion_matrix_2D \ div_RHS;
q = reshape(q, size(div));

q_grad_x, q_grad_y = gradfunc(q);

u_int_pred = u_int - q_grad_x;
v_int_pred = v_int - q_grad_y;

div_pred = divfunc(u_int_pred, v_int_pred);

print("Mean divergence: ", mean(abs.(div_pred)))





















# Solve possion for q with div as RHS using Gauss Seidel
# for i = 1:50000

#     q_diff_xx = circshift(q, (1, 0)) + circshift(q, (-1, 0))
#     q_diff_yy = circshift(q, (0, 1)) + circshift(q, (0, -1))

#     q_new = (q_diff_xx + q_diff_yy - dx * dx * div) / 4

#     dq = q_new - q
#     @. q += dq

#     print(sum(abs.(dq)), "\n")

# end

# function f(q, div)
#     laplacefunc(q) - div
# end
# prob = NonlinearProblem(f, q0, div)
# q = solve(prob, NewtonRaphson())



q_grad_x, q_grad_x = gradfunc(q)

u_int_pred = u_int - q_grad_x;
v_int_pred = v_int - q_grad_y;

div_pred = divfunc(u_int_pred, v_int_pred);

print("Mean divergence: ", mean(abs.(div_pred)))









































using Statistics
using Plots

function meshgrid(xin,yin)
    nx=length(xin)
    ny=length(yin)
    xout=zeros(ny,nx)
    yout=zeros(ny,nx)
    for jx=1:nx
        for ix=1:ny
            xout[ix,jx]=xin[jx]
            yout[ix,jx]=yin[ix]
        end
    end
    return (x=xout, y=yout)
    end


x = LinRange(0, 1, 64);
y = LinRange(0, 1, 64);

x, y = meshgrid(x, y);

f(x, y) = sin.(2*pi .* x) + cos.(2*pi .* y);

z = f(x, y);

heatmap(z)




using FFTW

function create_params(
    K;
    nu,
    f = zeros(2K, 2K),
    m = nothing,
    θ = nothing,
    anti_alias_factor = 2 / 3,
)
    Kf = round(Int, anti_alias_factor * K)
    N = 2K
    x = LinRange(0.0f0, 1.0f0, N+1)[2:end]

    # Vector of wavenumbers

    k = ArrayType(fftfreq(N, Float32(N)))
    normk = k .^ 2 .+ k' .^ 2

    # Projection components
    kx = k
    ky = reshape(k, 1, :)
    Pxx = @. 1 - kx * kx / (kx^2 + ky^2)
    Pxy = @. 0 - kx * ky / (kx^2 + ky^2)
    Pyy = @. 1 - ky * ky / (kx^2 + ky^2)

    # The zeroth component is currently `0/0 = NaN`. For `CuArray`s,
    # we need to explicitly allow scalar indexing.

    CUDA.@allowscalar Pxx[1, 1] = 1
    CUDA.@allowscalar Pxy[1, 1] = 0
    CUDA.@allowscalar Pyy[1, 1] = 1

    # Closure model
    m = nothing
    θ = nothing

    (; x, N, K, Kf, k, nu, normk, f, Pxx, Pxy, Pyy, m, θ)
end

function project(u, params)
    (; Pxx, Pxy, Pyy) = params
    ux, uy = eachslice(u; dims = 3)
    dux = @. Pxx * ux + Pxy * uy
    duy = @. Pxy * ux + Pyy * uy
    cat(dux, duy; dims = 3)
end


params = create_params(32; nu = 0.001f0)

u = trainset[:, :, 1:2, 10, 1];



u_hat = fft(u, (1, 2));

u_hat_project = project(u_hat, params);
maximum(abs, params.k .* u_hat_project[:, :, 1] .+ params.k' .* u_hat_project[:, :, 2])

u_proj = real.(ifft(u_hat_project, (1, 2)));

heatmap(u[:, :, 2])
heatmap(u_proj[:, :, 2])

u_int = u[:, :, 1];
v_int = u[:, :, 2];

dx = 1/(size(u_int, 1)-1);
dy = 1/(size(v_int, 1)-1);

u_diff_x = (u_int - circshift(u_int, (1, 0))) / dx;
v_diff_y = (v_int - circshift(v_int, (0, 1))) / dy;


div = u_diff_x + v_diff_y;
print("Mean divergence: ", mean(abs.(div)))










params = create_params(32; nu = 0.001f0)

u = trainset[:, :, 1:2, 10, 1];



u_int = trainset[:, :, 1, 10, 1];
v_int = trainset[:, :, 2, 10, 1];

u_int = u_int + z;
v_int = v_int - z;

u = cat(u_int, v_int; dims = 3);

u = project(u, params);

u_int = u[:, :, 1];
v_int = u[:, :, 2];


dx = 1/(size(u_int, 1)-1);
dy = 1/(size(v_int, 1)-1);

u_diff_x = (u_int - circshift(u_int, (1, 0))) / dx;
v_diff_y = (v_int - circshift(v_int, (0, 1))) / dy;

div = u_diff_x + v_diff_y;
print("Mean divergence: ", mean(abs.(div)))




lol = fft(u)
u = project(lol, params);
u = ifft(u)













using LinearAlgebra

u_int = u_int + z;
v_int = v_int - z;

heatmap(u_int)

dx = 1/(size(u_int, 1)-1);
dy = 1/(size(v_int, 1)-1);

u_diff_x = (u_int - circshift(u_int, (1, 0))) / dx;
v_diff_y = (v_int - circshift(v_int, (0, 1))) / dy;

div = u_diff_x + v_diff_y;
print("Mean divergence: ", mean(abs.(div)))

div_RHS = reshape(div, size(div)[1]*size(div)[2]);

possion_matrix_1D = Tridiagonal(
    ones(size(div, 1)-1),
    -4*ones(size(div, 1)),
    ones(size(div, 1)-1)
);
possion_matrix_1D = Matrix(possion_matrix_1D);
possion_matrix_1D[1, end] = 1;
possion_matrix_1D[end, 1] = 1;

S = Tridiagonal(
    ones(size(div, 1)-1),
    zeros(size(div, 1)),
    ones(size(div, 1)-1)
);
S = Matrix(S);


possion_matrix_2D = kron(I(size(div, 2)), possion_matrix_1D) + kron(S, I(size(div, 2)));
possion_matrix_2D = possion_matrix_2D / (dx * dx);

q = possion_matrix_2D \ div_RHS;
q = reshape(q, size(div));


q_grad = zeros(size(u_int, 1), size(u_int, 2), 2);
q_grad[:, :, 1] = (q - circshift(q, (1, 0))) / dx;
q_grad[:, :, 2] = (q - circshift(q, (0, 1))) / dy;

u_int_pred = u_int - q_grad[:, :, 1];
v_int_pred = v_int - q_grad[:, :, 2];

u_diff_x_pred = (u_int_pred - circshift(u_int_pred, (1, 0))) / dx;
v_diff_y_pred = (v_int_pred - circshift(v_int_pred, (0, 1))) / dy;

div_pred = u_diff_x_pred + v_diff_y_pred;

print("Mean divergence: ", mean(abs.(div_pred)))

heatmap(u_int_pred)





possion_matrix = zeros(size(div_RHS, 1), size(div_RHS, 1));

for i = 1:size(div_RHS, 1)
    possion_matrix[i, i] = 4;
    if i + 1 <= size(div_RHS, 1)
        possion_matrix[i, i+1] = -1;
    end
    if i - 1 >= 1
        possion_matrix[i, i-1] = -1;
    end
    if i + size(div_RHS, 1) <= size(div_RHS, 1)
        possion_matrix[i, i+size(div_RHS, 1)] = -1;
    end
    if i - size(div_RHS, 1) >= 1
        possion_matrix[i, i-size(div_RHS, 1)] = -1;
    end
end


# periodic boundary conditions for possion matrix
for i = 1:size(div_RHS, 1)
    if i % size(div_RHS, 1) == 0
        possion_matrix[i, i-1] = -1;
        possion_matrix[i, i-size(div_RHS, 1)] = -1;
        possion_matrix[i, i+1-size(div_RHS, 1)] = -1;
    end
    if i % size(div_RHS, 1) == 1
        possion_matrix[i, i+1] = -1;
        possion_matrix[i, i+size(div_RHS, 1)] = -1;
        possion_matrix[i, i-1+size(div_RHS, 1)] = -1;
    end
end

q = possion_matrix \ div_RHS;

q = reshape(q, size(div));



q_grad = zeros(size(u_int, 1), size(u_int, 2), 2)
q_grad[:, :, 1] = (q - circshift(q, (1, 0))) / cell_length_x;
q_grad[:, :, 2] = (q - circshift(q, (0, 1))) / cell_length_y;

u_int_pred = u_int - q_grad[:, :, 1];
v_int_pred = v_int - q_grad[:, :, 2];

u_diff_x_pred = (u_int_pred - circshift(u_int_pred, (1, 0))) / cell_length_x;
v_diff_y_pred = (v_int_pred - circshift(v_int_pred, (0, 1))) / cell_length_y;

div_pred = u_diff_x_pred + v_diff_y_pred;

print("Mean divergence: ", mean(abs.(div_pred)))






















# Compute the gradient of q
grad_q = zeros(size(u_int, 1), size(u_int, 2), 2)
grad_q[:, :, 1] = (q[3:end, :] - q[1:end-2, :]) / (2 * cell_length_x)
grad_q[:, :, 2] = (q[:, 3:end] - q[:, 1:end-2]) / (2 * cell_length_y)


div_free_u = u_int - grad_q[:, :, 1]
div_free_v = v_int - grad_q[:, :, 2]

u_int = div_free_u
v_int = div_free_v

div = zeros(size(u_int))
div[2:end-1, 2:end-1] = (u_int[2:end-1, 2:end-1] - u_int[1:end-2, 2:end-1]) / cell_length_x +
      (v_int[2:end-1, 2:end-1] - v_int[2:end-1, 1:end-2]) / cell_length_y

# Periodic boundary conditions
div[1, :] = u_int[1, :] - u_int[end-1, :];
div[end, :] = u_int[2, :] - u_int[end, :];
div[:, 1] = v_int[:, 1] - v_int[:, end-1];
div[:, end] = v_int[:, 2] - v_int[:, end];

print("Mean divergence: ", mean(abs.(div)))





















num_target = size(trainset_target_distribution)[end]
train_ids = shuffle(rng, 1:num_target)
trainset_target_distribution = trainset_target_distribution[:, :, :, train_ids]
trainset_init_distribution = trainset_init_distribution[:, :, :, :, train_ids]
trainset_pars_distribution = trainset_pars_distribution[:, train_ids]

x_1 = trainset_target_distribution[:, :, :, 1:50]
x_0 = trainset_init_distribution[:, :, :, end, 1:50]

t = rand!(rng, similar(x_1, 1, 1, 1, 50))

interpolant = Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt);

I = interpolant.interpolant(x_0, x_1, t) .|> dev
dI_dt = interpolant.dinterpolant_dt(x_0, x_1, t) .|> dev



# Compute divergence
using Statistics

for i = 1:50
    I = I

    z = randn!(rng, similar(x_1, size(x_1)))
    z = sqrt.(t) .* z

    g = interpolant.gamma(t)

    print(g[1, 1, 1, 1:1] .* z[1, 1, 1, 1:1])

    I = I .+ g .* z;

    u_int = I[:, :, 1, i];
    v_int = I[:, :, 2, i];

    cell_length_x = 1/size(u_int, 1)
    cell_length_y = 1/size(v_int, 2)

    divergence = (u_int[2:end-1, 2:end-1] - u_int[1:end-2, 2:end-1]) / cell_length_x +
                (v_int[2:end-1, 2:end-1] - v_int[2:end-1, 1:end-2]) / cell_length_y

    mean_divergence = mean(abs.(divergence))
    println("Mean divergence: ", mean_divergence, "id: ", i)
end;



using Tullio
using CUDA


join_heads(x) = reshape(x, :, size(x)[3:end]...)

C = 16;
H = 4;
N = 1024;
B = 2;

q = convert(Array{T}, randn(rng, (C, H, N, B)));
k = convert(Array{T}, randn(rng, (C, H, N, B)));
v = convert(Array{T}, randn(rng, (C, H, N, B)));

v = permutedims(v, (3, 1, 2, 4)) # (C, H, N, B) -> (N, C, H, B)
kt = permutedims(k, (1, 3, 2, 4)) # (C, H, N, B) -> (C, N, H, B)

x = batched_mul(kt, v) # (C, N, H, B) * (N, C, H, B) -> (C, C, H, B)

q = permutedims(q, (3, 1, 2, 4)) # (C, H, N, B) -> (N, C, H, B)

x = batched_mul(q, x) # (N, C, H, B) * (C, C, H, B) -> (N, C, H, B)

x = permutedims(x, (2, 3, 1, 4)) # (N, C, H, B) -> (C, H, N, B)



##### Test stochastic interpolant #####
st_ = Lux.testmode(st);
gif_save_path = "output/sde_SI";
num_test_paths = 5;


compare_sde_pred_with_true(
    model,
    ps,
    st_,
    testset,
    testset_pars,
    num_test_paths,
    normalize_data,
    mask,
    5,
    gif_save_path,
    rng,
    dev,
)

for num_generator_steps = 45:50;
    print("Number of generator steps: ", num_generator_steps)
    print("\n") 
    compare_sde_pred_with_true(
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
end;



# function transform_to_nothing(x::AbstractArray)
#     return nothing
# end


# x = randn(rng, (H, W, C, 2, 5))
# y =  randn(rng, (H, W, C, 2, 5))
# y = transform_to_nothing(x)


# lol = pars_cat(x, y; dims=1)



# function pars_cat(x::AbstractArray, y::Nothing; dims=1)
#     return x
# end

# function pars_cat(x::AbstractArray, y::AbstractArray; dims=1)
#     return cat(x, y; dims=dims)
# end


























data_train = load("data/data_train_large.jld2", "data_train");




data_train = load("data/turbulence.jld2", "data_train");




    
start_time, num_steps, skip_steps = time_step_info

start_time = 1;
num_steps = 400;
skip_steps = 1;

state_dims = (64, 64, 2);

data_train = load("data/data_train.jld2", "data_train");
trainset_state = zeros(state_dims...,  num_steps, 5);
for i in 1:5
    for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
        trainset_state[:, :, 1, j, i] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
        trainset_state[:, :, 2, j, i] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]
    end
end

trainset_pars = zeros(num_pars, num_steps, length(data_ids));





















# test_init_condition = testset[:, :, :, 1:1, 1:1] |> dev;
# test_pars = testset_pars[:, 1:1, :] |> dev;
# num_test_steps = size(testset, 4) |> dev;
num_test_paths = 5 |> dev;


num_test_trajectories = size(testset)[end];
num_channels = size(testset, 3);
num_test_steps = size(testset, 4);

st_ = Lux.testmode(st);

if !isnothing(normalize_data)
    x_true = normalize_data.inverse_transform(testset)
else
    x_true = testset
end;

if !isnothing(mask)
    x_true = x_true .* mask
    num_non_obstacle_grid_points = sum(mask)
else
    num_non_obstacle_grid_points = size(x_true)[1] * size(x_true)[2]
end;

pathwise_MSE = []
mean_MSE = []
x = zeros(H, W, C, num_test_steps, num_test_paths) |> dev;
for i = 1:num_test_trajectories

    test_init_condition = testset[:, :, :, 1:1, i]
    test_pars = testset_pars[:, 1:1, i]

    x = compute_multiple_SDE_steps(
        init_condition=test_init_condition,
        parameters=test_pars,
        num_physical_steps=num_test_steps,
        num_generator_steps=40,
        num_paths=num_test_paths,
        model=model,
        ps=ps,
        st=st_,
        rng=rng,
        dev=dev,
        mask=mask,
    )


    if !isnothing(normalize_data)
        x = normalize_data.inverse_transform(x)
    end

    if !isnothing(mask)
        x = x .* mask
    end

    error_i = 0
    for j = 1:num_test_paths
        error_i += sum((x[:, :, :, :, j] - x_true[:, :, :, :, i]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
    end
    error_i = error_i / num_test_paths

    push!(pathwise_MSE, error_i)

    x_mean = mean(x, dims=5)[:, :, :, :, 1]
    x_std = std(x, dims=5)[:, :, :, :, 1]

    MSE = sum((x_mean - x_true[:, :, :, :, i]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
    push!(mean_MSE, MSE)
end;

println("Mean of pathwise MSE: ", mean(pathwise_MSE))
println("Std of pathwise MSE: ", std(pathwise_MSE))

println("Mean of mean MSE (SDE): ", mean(mean_MSE))
println("Std of mean MSE (SDE): ", std(mean_MSE))

x_mean = mean(x, dims=5)[:, :, :, :, 1];
x_std = std(x, dims=5)[:, :, :, :, 1];

x_true = x_true[:, :, :, :, num_test_trajectories];

save_path = @sprintf("output/tra_long_sde_SI_test.gif")

preds_to_save = (x_true[:, :, 4, :], x_mean[:, :, 4, :], Float16.(x_mean[:, :, 4, :]-x_true[:, :, 4, :]), Float16.(x_std[:, :, 4, :]), x[:, :, 4, :, 1], x[:, :, 4, :, 2], x[:, :, 4, :, 3], x[:, :, 4, :, 4]);
create_gif(preds_to_save, save_path, ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"])

CUDA.reclaim()
GC.gc()



# x = compute_multiple_ODE_steps(
#     init_condition=test_init_condition,
#     parameters=test_pars,
#     num_physical_steps=num_test_steps,
#     num_generator_steps=25,
#     model=model,
#     ps=ps,
#     st=st,
#     dev=dev,
#     mask=mask
# );

# num_channels = size(x, 3)

# if !isnothing(normalize_data)
#     x = normalize_data.inverse_transform(x)
#     testset = normalize_data.inverse_transform(testset)
# end

# if !isnothing(mask)
#     x = x .* mask
#     testset = testset .* mask

#     num_non_obstacle_grid_points = sum(mask)
# end

# MSE = sum((x[:, :, :, :, 1] - testset[:, :, :, :, 1]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
# println("MSE (ODE): ", MSE)

# # println("Time stepping error (ODE): ", mean(error))

# x = x[:, :, 4, :, 1]
# x_true = testset[:, :, 4, :, 1]


# save_path = @sprintf("output/ode_SI_%i.gif", 1)

# preds_to_save = (x_true, x, x-x_true)
# create_gif(preds_to_save, save_path, ["True", "Pred", "Error"])



# x = compute_multiple_SDE_steps(
#     init_condition=test_init_condition,
#     parameters=test_pars,
#     num_physical_steps=num_test_steps,
#     num_generator_steps=25,
#     num_paths=num_test_paths,
#     model=model,
#     ps=ps,
#     st=st,
#     rng=rng,
#     dev=dev,
#     mask=mask
# );


# num_channels = size(x, 3)
                
# if !isnothing(normalize_data)
#     x = normalize_data.inverse_transform(x)
#     testset = normalize_data.inverse_transform(testset)
# end

# if !isnothing(mask)
#     x = x .* mask
#     testset = testset .* mask

#     num_non_obstacle_grid_points = sum(mask)
# end

# MSE = 0.0f0
# for i = 1:num_test_paths
#     MSE += sum((x[:, :, :, :, i] - testset[:, :, :, :, 1]).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
# end

# MSE /= num_test_paths
# println("MSE over paths (SDE): ", MSE)

# x_true = testset[:, :, :, :, 1]

# x_mean = mean(x, dims=5)[:, :, :, :, 1]
# x_std = std(x, dims=5)[:, :, :, :, 1]

# MSE = sum((x_mean - x_true).^2) / num_non_obstacle_grid_points / num_test_steps / num_channels
# println("MSE over mean (SDE): ", MSE)

# save_path = @sprintf("output/sde_SI_%i.gif", 1)

# preds_to_save = (x_true[:, :, 4, :], x_mean[:, :, 4, :], Float16.(x_mean[:, :, 4, :]-x_true[:, :, 4, :]), Float16.(x_std[:, :, 4, :]), x[:, :, 4, :, 1], x[:, :, 4, :, 2], x[:, :, 4, :, 3], x[:, :, 4, :, 4])
# create_gif(preds_to_save, save_path, ["True", "Pred mean", "Error", "Pred std", "Pred 1", "Pred 2", "Pred 3", "Pred 4"])



# x = trainset_init_distribution[:, :, :, 1:4] |> dev;
# x0 = x  |> dev;
# pars = randn(rng, (num_pars, 4)) |> dev;
# t = rand(rng, Float32, (1, 1, 1, 4)) |> dev;

# out, st = model.velocity((x, x0, pars, t), ps, st);



# if "inc" in l and "mixed" in l:
#     # ORDER (fields): velocity (x,y), --, pressure, ORDER (params): rey, --, --
#     self.normMean = np.array([0.444969, 0.000299, 0, 0.000586, 550.000000, 0, 0], dtype=np.float32)
#     self.normStd =  np.array([0.206128, 0.206128, 1, 0.003942, 262.678467, 1, 1], dtype=np.float32)

# if "tra" in l and "mixed" in l:
#     # ORDER (fields): velocity (x,y), density, pressure, ORDER (params): rey, mach, --
#     self.normMean = np.array([0.560642, -0.000129, 0.903352, 0.637941, 10000.000000, 0.700000, 0], dtype=np.float32)
#     self.normStd =  np.array([0.216987, 0.216987, 0.145391, 0.119944, 1, 0.118322, 1], dtype=np.float32)

# if "iso" in l and "single" in l:
#     # ORDER (fields): velocity (x,y,z), pressure, ORDER (params): --, --, --
#     self.normMean = np.array([-0.054618, -0.385225, -0.255757, 0.033446, 0, 0, 0], dtype=np.float32)
#     self.normStd =  np.array([0.539194, 0.710318, 0.510352, 0.258235, 1, 1, 1], dtype=np.float32)





###### load training set #####
# trainset = MNIST(:train)
# # trainset = CIFAR10(:train)
# trainset = trainset[1:num_train];
# trainset = trainset.features;
# trainset = imresize(trainset, (32, 32, num_train));
# trainset = reshape(trainset, 32, 32, 1, num_train);

# start_time = 100;
# num_steps = 150;
# num_train = 5;
# skip_steps = 2;
# data_train = load("data/data_train.jld2", "data_train");
# #data_train[1].data[1].u[200][2]
# H, W = size(data_train[1].data[1].u[1][1]).-2;
# C = 2;
# trainset = zeros(H, W, C, num_steps, num_train); # Height, Width, Channels, num_steps, num_train
# for i in 1:num_train
#     counter = 1;
#     for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
#         trainset[:, :, 1, counter, i] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
#         trainset[:, :, 2, counter, i] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]

#         trainset[:, :, :, counter, i] = trainset[:, :, :, counter, i] ./ norm(trainset[:, :, :, counter, i]);

#         counter += 1
#     end
# end

# trainset[:, :, 1, :, :] = (trainset[:, :, 1, :, :] .- mean(trainset[:, :, 1, :, :])) ./ std(trainset[:, :, 1, :, :]);
# trainset[:, :, 2, :, :] = (trainset[:, :, 2, :, :] .- mean(trainset[:, :, 2, :, :])) ./ std(trainset[:, :, 2, :, :]);

# # min max normalization
# # trainset = (trainset .- minimum(trainset)) ./ (maximum(trainset) - minimum(trainset));

# x = sqrt.(trainset[:, :, 1, :, 1].^2 + trainset[:, :, 2, :, 1].^2);

# create_gif((x, x), "HF.gif", ("lol", "lol"))










# velocity = ConditionalUNet(
#     image_size; 
#     in_channels=in_channels,
#     channels=channels, 
#     block_depth=block_depth,
#     min_freq=min_freq, 
#     max_freq=max_freq, 
#     embedding_dims=embedding_dims,
# )
# velocity =  ConvNextUNet(
#     image_size; 
#     in_channels=in_channels,
#     channels=channels, 
#     block_depth=block_depth,
#     min_freq=min_freq, 
#     max_freq=max_freq, 
#     embedding_dims=embedding_dims
# )
# velocity = DitParsConvNextUNet(
#     image_size; 
#     in_channels=in_channels,
#     channels=channels, 
#     block_depth=block_depth,
#     min_freq=min_freq, 
#     max_freq=max_freq, 
#     embedding_dims=embedding_dims,
#     pars_dim=1
# )
# velocity = ConditionalDiffusionTransformer(
#     image_size;
#     in_channels=in_channels, 
#     patch_size=(8, 8),
#     embed_dim=256, 
#     depth=4, 
#     number_heads=8,
#     mlp_ratio=4.0f0, 
#     dropout_rate=0.1f0, 
#     embedding_dropout_rate=0.1f0,
# )