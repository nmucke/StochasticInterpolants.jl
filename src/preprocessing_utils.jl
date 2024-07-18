


function expand_dims(
    data,
    mean,
    std,
)
    
    num_dims = length(size(data))

    if num_dims == 3
        mean = reshape(mean, 1, 1, 4)
        std = reshape(std, 1, 1, 4)
    elseif num_dims == 4
        mean = reshape(mean, 1, 1, 4, 1)
        std = reshape(std, 1, 1, 4, 1)
    elseif num_dims == 5
        mean = reshape(mean, 1, 1, 4, 1, 1)
        std = reshape(std, 1, 1, 4, 1, 1)
    end

    return mean, std
end

"""
    NormalizeData(mean, std)

A container for the normalization and inverse normalization functions
"""
struct NormalizeData
    transform::Function
    inverse_transform::Function
end

function NormalizeData(
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

    return NormalizeData(transform, inverse_transform)
    



end
