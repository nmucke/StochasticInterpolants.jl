

"""
    Gamma
    A struct for the gamma function and its derivative
"""
struct Gamma
    gamma
    dgamma_dt
end

function Gamma(
    gamma::Function = t -> 1f0 .- t,
    dgamma_dt::Function = t -> -ones(size(t)),
)

    diffusion_multiplier = diffusion_multiplier

    gamma(t) = diffusion_multiplier .* gamma(t)
    dgamma_dt(t) = diffusion_multiplier * dgamma_dt

    return Gamma(gamma, dgamma_dt)
end

"""
    DiffusionCoefficient
    A struct for the diffusion coefficient function
"""
struct DiffusionCoefficient
    diffusion_coefficient
end

function DiffusionCoefficient(
    diffusion_coefficient::Function = t -> sqrt.((3f0 .- t) .* (1f0 .- t))
)

    return DiffusionCoefficient(diffusion_coefficient)
end
