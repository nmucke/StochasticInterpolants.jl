

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
    dgamma_dt = -ones(size(t))
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
