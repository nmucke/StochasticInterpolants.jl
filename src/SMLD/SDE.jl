using Lux
using Random
using CUDA
using NNlib
using Setfield

"""
    drift_term(
        x::AbstractArray,
        t::AbstractArray
    )

Drift term, f(x,t)=0, for the forward process
"""
function drift_term(
    x::AbstractArray,
    t::AbstractArray
)

    return zeros(size(x))
end

"""
    diffusion_term(
        x::AbstractArray,
        t::AbstractArray;
        dsigma_squared_dt::Function = nothing,
        alpha::Function = nothing
    )

Diffusion term, g(x,t)=\alpha(t)=sqrt(d/dt [\sigma(t)^2]), for the forward process.
- If the time derivative of dsigma_squared_dt is provided sqrt(dsigma_squared_dt(t)) is used.
- If alpha is provided, alpha(t) is used.
"""
function diffusion_term(
    x::AbstractArray,
    t::AbstractArray;
    dsigma_squared_dt::Function = nothing,
    alpha::Function = nothing
)

    if dsigma_squared_dt == nothing
        diffusion(t) = alpha(t)
    elseif isnothing(alpha)


    return zeros(size(x))
end


"""
    marginal_probability_std(
        t::AbstractArray,
        sigma::AbstractFloat,
    )

Computes the standard deviation of the marginal probability of the noise

Based on https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=LZC7wrOvxLdL
"""
function marginal_probability_std(
    t::AbstractArray,
    sigma::AbstractFloat,
)

    return sqrt.((sigma.^(2 .* t) .- 1) ./ 2 ./ log.(sigma))
end

"""
    diffusion_coefficient(
        t::AbstractArray,
        sigma::AbstractFloat,
    )

Computes the diffusion coefficient of the noise

Based on https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=LZC7wrOvxLdL
"""
function diffusion_coefficient(
    t::AbstractArray,
    sigma::AbstractFloat,
)
    return sigma.^t
end

