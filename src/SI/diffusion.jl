


function get_diffusion_coefficient(
    type::String,
    multiplier,
)
    if type == "linear"
        diffusion_coefficient = t -> multiplier .* (1f0 .- t);
    elseif type == "follmer_optimal"
        diffusion_coefficient = t -> multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t));
    end
    return diffusion_coefficient
end
