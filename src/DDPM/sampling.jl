using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA



"""
    sample_timestep(x, t)

Calls the model to predict the noise in the image and returns 
the denoised image. 
Applies noise to this image, if we are not in the last step yet.
"""
function sample_timestep(
    x, 
    t,
    model,
    noise_scheduling,
    ps,
    st,
    rng,
    dev=gpu
)


    if length(size(t)) == 4
        
        # Call model (current image - noise prediction)
        noise_pred, st = model((x, t), ps, st)

        t = t[1, 1, 1, :]

    else

        # Call model (current image - noise prediction)
        noise_pred, st = model((x, reshape(t, 1, 1, 1, length(t))), ps, st)

    end

    # Get noise scheduling parameters
    betas_t = noise_scheduling.betas[t]
    sqrt_one_minus_alphas_cumprod_t = noise_scheduling.sqrt_one_minus_alphas_cumprod[t]    
    sqrt_recip_alphas_t = noise_scheduling.sqrt_recip_alphas[t]
    posterior_variance_t = noise_scheduling.posterior_variance[t]

    # Reshape noise scheduling parameters
    betas_t = reshape(betas_t, (1, 1, 1, size(x)[end]))
    sqrt_one_minus_alphas_cumprod_t = reshape(sqrt_one_minus_alphas_cumprod_t, (1, 1, 1, size(x)[end]))
    sqrt_recip_alphas_t = reshape(sqrt_recip_alphas_t, (1, 1, 1, size(x)[end]))
    posterior_variance_t = reshape(posterior_variance_t, (1, 1, 1, size(x)[end]))
    
    model_mean = sqrt_recip_alphas_t .* (
        x - betas_t .*  noise_pred ./ sqrt_one_minus_alphas_cumprod_t
    )
    
    if t == 0
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean, st
    else
        noise = randn(rng, size(x)) |> dev
        return model_mean .+ sqrt.(posterior_variance_t) .* noise, st
    end
end



