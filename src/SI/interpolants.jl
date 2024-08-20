

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

struct Interpolant
    alpha::Function
    beta::Function
    dalpha_dt::Function
    dbeta_dt::Function
    interpolant::Function
    dinterpolant_dt::Function
    gamma::Function
    dgamma_dt::Function
end


gamma = t -> diffusion_multiplier.* (1f0 .- t);
dgamma_dt = t -> -ones(size(t)) .* diffusion_multiplier;
diffusion_coefficient = t -> diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));


function Interpolant(
    alpha::Function = t -> 1f0 .- t,
    beta::Function = t -> t.^2,
    dalpha_dt::Function = t -> -1f0,
    dbeta_dt::Function = t -> 2f0 .* t,
    gamma::Function = t -> 1f0 .- t,
    dgamma_dt::Function = -ones(size(t))

)

    interpolant(x_0, x_1, t) = alpha(t) .* x_0 .+ beta(t) .* x_1
    dinterpolant_dt(x_0, x_1, t) = dalpha_dt(t) .* x_0 + dbeta_dt(t) .* x_1

    return Interpolant(alpha, beta, dalpha_dt, dbeta_dt, interpolant, dinterpolant_dt, gamma, dgamma_dt)
end

# """
#     linear_interpolant(x_0, x_1, t)

# Interpolates between two arrays `x_0` and `x_1` using a linear interpolant. 
# The interpolation is parameterized by `t` which is a scalar. 
# The output is an array of the same shape as `x_0` and `x_1`.
# """
# function linear_interpolant(
#     x_0::AbstractArray,
#     x_1::AbstractArray,
#     t::AbstractArray,
# )   

#     I = (1 .- t) .* x_0 .+ t .* x_1
        
#     dI_dt = x_1 .- x_0

#     return I, dI_dt
# end