


"""
    linear_interpolant(x_0, x_1, t)

Interpolates between two arrays `x_0` and `x_1` using a linear interpolant. 
The interpolation is parameterized by `t` which is a scalar. 
The output is an array of the same shape as `x_0` and `x_1`.
"""
function linear_interpolant(
    x_0::AbstractArray,
    x_1::AbstractArray,
    t::AbstractArray,
    return_time_derivative::Bool=false,
)   

    I = (1 .- t) .* x_0 .+ t .* x_1

    if return_time_derivative
        
        dI_dt = x_1 .- x_0

        return I, dI_dt
    end
    
    return I
end