

"""
    Gamma
    A struct for the gamma function and its derivative
"""
struct Gamma
    gamma::Function
    dgamma_dt::Function
end

"""
    DiffusionCoefficient
    A struct for the diffusion coefficient function
"""
struct DiffusionCoefficient
    diffusion_coefficient
end