using Lux
using Random
using CUDA
using NNlib
using Setfield


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

