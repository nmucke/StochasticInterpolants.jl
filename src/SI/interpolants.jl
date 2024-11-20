

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
# struct Interpolant{A,B,C,D,E,F,G,H}
#     alpha::Function
#     beta::B
#     dalpha_dt::C
#     dbeta_dt::D
#     interpolant::E
#     dinterpolant_dt::F
#     gamma::G
#     dgamma_dt::H
# end

# struct Interpolant
#     alpha::Function
#     beta::Function
#     dalpha_dt::Function
#     dbeta_dt::Function
#     interpolant::Function
#     dinterpolant_dt::Function
#     gamma::Function
#     dgamma_dt::Function
# end

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
