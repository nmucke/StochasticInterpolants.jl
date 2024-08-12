


function expand_dims(
    data, mean, std,
)
    
    num_dims = length(size(data))
    num_channels = size(data, 3)

    if num_dims == 3
        mean = reshape(mean, 1, 1, num_channels)
        std = reshape(std, 1, 1, num_channels)
    elseif num_dims == 4
        mean = reshape(mean, 1, 1, num_channels, 1)
        std = reshape(std, 1, 1, num_channels, 1)
    elseif num_dims == 5
        mean = reshape(mean, 1, 1, num_channels, 1, 1)
        std = reshape(std, 1, 1, num_channels, 1, 1)
    end

    return mean, std
end

"""
    StandardizeData(mean, std)

A container for the standardization and inverse standardization functions.
Data is transform to have zero mean and unit variance.
"""
struct StandardizeData
    transform::Function
    inverse_transform::Function
end

function StandardizeData(
    mean::AbstractArray, 
    std::AbstractArray
)
    transform(x) = begin
        _mean, _std = expand_dims(x, mean, std)
        return (x .- _mean) ./ _std        
    end

    inverse_transform(x) = begin
        _mean, _std = expand_dims(x, mean, std)
        return x .* _std .+ _mean
    end

    return StandardizeData(transform, inverse_transform)
    
end

function expand_dims_pars(
    data, min, max,
)
    
    num_dims = length(size(data))
    num_pars = size(data, 1)

    if num_dims == 2
        min = reshape(min, num_pars, 1)
        max = reshape(max, num_pars, 1)
    elseif num_dims == 3
        min = reshape(min, num_pars, 1, 1)
        max = reshape(max, num_pars, 1, 1)
    end

    return min, max
end

"""
NormalizePars(mean, std)

A container for the normalization and inverse normalization functions.
Data is transformed to be between 0 and 1.
"""
struct NormalizePars
    transform::Function
    inverse_transform::Function
end

function NormalizePars(
    min::AbstractArray, 
    max::AbstractArray
)
    transform(x) = begin
        _min, _max = expand_dims_pars(x, min, max)
        return (x .- _min) ./ (_max - _min)
    end

    inverse_transform(x) = begin
        _min, _max = expand_dims_pars(x, min, max)
        return x .* (_max - _min) .+ _min
    end

    return NormalizePars(transform, inverse_transform)
    
end


