


pif0 = Float32(pi)

"""
    Interpolant(
        alpha, 
        beta, 
        dalpha_dt, 
        dbeta_dt, 
        interpolant, 
        dinterpolant_dt
    )


A struct that contains the functions for the interpolation and its derivatives.
"""
function Interpolant(
    alpha = t -> 1f0 .- t,
    beta = t -> t.^2,
    dalpha_dt = t -> -1f0,
    dbeta_dt = t -> 2f0 .* t,
    gamma = t -> 1f0 .- t,
    dgamma_dt = t -> -ones(size(t))
)

    interpolant(x_0, x_1, t) = alpha(t) .* x_0 .+ beta(t) .* x_1
    dinterpolant_dt(x_0, x_1, t) = dalpha_dt(t) .* x_0 + dbeta_dt(t) .* x_1

    return (; alpha, beta, dalpha_dt, dbeta_dt, interpolant, dinterpolant_dt, gamma, dgamma_dt)
end

function get_alpha(type::String)
    if type == "linear"
        alpha = t -> 1f0 .- t
        dalpha_dt = t -> -1f0
    elseif type == "quadratic"
        alpha = t -> 1f0 .- t.^2
        dalpha_dt = t -> -2f0 .* t
    end
    return alpha, dalpha_dt
end

function get_beta(type::String)
    if type == "linear"
        beta = t -> t
        dbeta_dt = t -> t
    elseif type == "quadratic"
        beta = t -> t.^2
        dbeta_dt = t -> 2f0 .* t
    end
    return beta, dbeta_dt
end

function get_gamma(type::String, multiplier)
    if type == "linear"
        gamma = t -> multiplier .* (1f0 .- t)
        dgamma_dt = t -> -1f0 .* multiplier
    elseif type == "quadratic"
        gamma = t -> multiplier .* (1f0 .- t.^2)
        dgamma_dt = t -> -2f0 .* t .* multiplier
    end
    return gamma, dgamma_dt
end

function get_interpolant(
    alpha_type::String,
    beta_type::String,
    gamma_type::String,
    gamma_multiplier,
)
    alpha, dalpha_dt = get_alpha(alpha_type)
    beta, dbeta_dt = get_beta(beta_type)
    gamma, dgamma_dt = get_gamma(gamma_type, gamma_multiplier)

    return Interpolant(alpha, beta, dalpha_dt, dbeta_dt, gamma, dgamma_dt)
end


function get_alpha_series(t, coefs)

    num_terms = length(coefs)
    sin_evals = [sin.(i .* pif0 .* t) for i in 1:num_terms]
    alpha_series = cos.(0.5f0 .* pif0 .* t) .+ 1f0 ./ num_terms .*  coefs'sin_evals
    
    return alpha_series
end

function get_dalpha_series_dt(t, coefs)

    num_terms = length(coefs)
    dsin_dt_evals = [i .* pif0 * cos.(i .* pif0 .* t) for i in 1:num_terms]
    dalpha_series_dt = -0.5f0 .* pif0 .* sin.(0.5f0 .* pif0 .* t) + 1f0 ./ num_terms .* coefs'dsin_dt_evals
    
    return dalpha_series_dt
end

function get_beta_series(t, coefs)
    
    num_terms = length(coefs)

    sin_evals = [sin.(i .* pif0 .* t) for i in 1:num_terms]

    beta_series = sin.(0.5f0 .* pif0 .* t) + 1f0 ./ num_terms .*   coefs'sin_evals
    
    return beta_series
end

function get_dbeta_series_dt(t, coefs)
    
    num_terms = length(coefs)
    dsin_dt_evals = [i .* pif0 * cos.(i .* pif0 .* t) for i in 1:num_terms]
    dbeta_series_dt = 0.5f0 .* pif0 .* cos.(0.5f0 .* pif0 .* t) + 1f0 ./ num_terms .* coefs'dsin_dt_evals
    
    return dbeta_series_dt
end

function get_gamma_series(t, coefs)

    num_terms = length(coefs)
    sin_evals = [sin.(i .* pif0 .* t) for i in 1:num_terms]
    gamma_series = cos.(0.5f0 .* pif0 .* t) + 1f0 ./ num_terms .*  coefs'sin_evals
    
    return gamma_series
end

function get_dgamma_series_dt(t, coefs)

    num_terms = length(coefs)
    dsin_dt_evals = [i .* pif0 * cos.(i .* pif0 .* t) for i in 1:num_terms]
    dgamma_series_dt = -0.5f0 .* pif0 .* sin.(0.5f0 .* pif0 .* t) + 1f0 ./ num_terms .* coefs'dsin_dt_evals
    
    return dgamma_series_dt
end

# coefs,
function d_interpolant_energy_dt(
    x_0,
    x_1,
    t,
    interpolant,
    omega=(0.04908f0, 0.04908f0),
    dev=gpu_device()
)

    H, W, C = size(x_0)[1], size(x_0)[2], size(x_0)[3]
    N = H * W * C

    alpha = interpolant.alpha(t) |> dev
    beta = interpolant.beta(t) |> dev
    gamma = interpolant.gamma(t) |> dev

    dalpha_dt = interpolant.dalpha_dt(t) |> dev
    dbeta_dt = interpolant.dbeta_dt(t) |> dev
    dgamma_dt = interpolant.dgamma_dt(t) |> dev


    x_0_norm = sum(x_0.^2, dims=(1, 2, 3)) * omega[1] * omega[2]
    x_1_norm = sum(x_1.^2, dims=(1, 2, 3)) * omega[1] * omega[2]
    inner_product = sum(x_0 .* x_1, dims=(1, 2, 3)) * omega[1] * omega[2]


    x_0_norm_term = dalpha_dt .* alpha .* x_0_norm
    x_1_norm_term = dbeta_dt .* beta .* x_1_norm

    mix_term = (dbeta_dt .* alpha .+ beta .* dalpha_dt) .* inner_product

    w_norm_term = dgamma_dt .* gamma .* t .* N  * omega[1] * omega[2]

    gamma_squared_term = 0.5f0 .* gamma.^2 .* N  * omega[1] * omega[2]

    return x_0_norm_term .+ x_1_norm_term .+ mix_term .+ w_norm_term .+ gamma_squared_term
end

function interpolant_velocity(
    x_0,
    x_1,
    t,
    interpolant,
    omega=(0.04908f0, 0.04908f0),
    dev=gpu_device()
)
    # z = randn!(similar(x_1, size(x_1)))
    # W = sqrt.(t) .* z
    # W = map(t -> sqrt.(t) .* z, t)

    dalpha_dt = interpolant.dalpha_dt(t) |> dev
    dbeta_dt = interpolant.dbeta_dt(t) |> dev

    return dalpha_dt .* x_0 .+ dbeta_dt .* x_1 #.+ interpolant.dgamma_dt(t) .* W

    # velocity = map((t, w) -> dalpha_dt(t) .* x_0 .+ dbeta_dt(t) .* x_1 .+ dgamma_dt(t) .* w, t, W)
    # velocity = map((t, w) -> dalpha_dt(t) .* x_0 .+ dbeta_dt(t) .* x_1, t, W)
    # return velocity
end

function objective_function(coefs, x_0, x_1)

    # t = rand(100)
    t = LinRange(0.0f0, 1.0f0, 100)

    dalpha_dt = t -> get_dalpha_series_dt(t, coefs[1, :])
    dbeta_dt = t -> get_dbeta_series_dt(t, coefs[2, :])
    dgamma_dt = t -> get_dgamma_series_dt(t, coefs[3, :])

    velocity = interpolant_velocity(t, x_0, x_1, dalpha_dt, dbeta_dt, dgamma_dt)

    velocity_energy = map(i -> mean(velocity[i].^2), range(1, length(t)))

    velocity_energy = mean(velocity_energy)

    return velocity_energy
end

function compute_physics_consistent_interpolant_coefficients(
    trainset,
    num_coefs
)

    return 0
end