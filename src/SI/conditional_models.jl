

"""
ConditionalStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
struct ConditionalStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:velocity, :score)
}
    velocity::UNet
    score::UNet
    sde_sample::Function
    ode_sample::Function
    interpolant::Function
    loss::Function
    gamma::Function
end

"""
ConditionalStochasticInterpolant(
        image_size::Tuple{Int, Int}; 
        in_channels::Int = 3,
        channels=[32, 64, 96, 128], 
        block_depth=2,
        min_freq=1.0f0,
        max_freq=1000.0f0,
        embedding_dims=32,
        num_steps=100,
    )
    
    Constructs a Stochastic Interpolant model
"""
function ConditionalStochasticInterpolant(
    image_size::Tuple{Int, Int}; 
    sde_enabled::Bool = true,
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0,
    max_freq=1000.0f0,
    embedding_dims=32,
    num_steps=100,
)

    velocity = UNet(
        image_size; 
        in_channels=in_channels,
        channels=channels, 
        block_depth=block_depth,
        min_freq=min_freq, 
        max_freq=max_freq, 
        embedding_dims=embedding_dims,
    )

    interpolant = StochasticInterpolants.linear_interpolant

    diffusion_coefficient(t) = 1.0f0
    gamma(t, return_derivative=false) = begin
        out = sqrt.(2 .* t .* (1 .- t))
        if return_derivative
            deriv = (1 .- 2 .* t) ./ sqrt.(- 2 .* t .* (t .- 1))
            return out, deriv
        end
        return out
    end

    # sde_sample(num_samples, ps, st, rng, dev) = begin
    #     println("SDE sampling is not enabled")
    #     return nothing
    # end

    if sde_enabled

        score = UNet(
            image_size; 
            in_channels=in_channels,
            channels=channels, 
            block_depth=block_depth,
            min_freq=min_freq, 
            max_freq=max_freq, 
            embedding_dims=embedding_dims,
        )


        sde_sample(x_0, ps, st, dev) = StochasticInterpolants.sde_sampler(
            x_0, velocity, score, diffusion_coefficient, gamma, ps, st, num_steps, dev
        )

        # Loss including the score network
        loss(x_0, x_1, t, ps, st, rng, dev) = get_loss(
            x_0, x_1, t, velocity, score, interpolant, gamma, ps, st, rng, dev
        )

    else 

        # Loss without the score network
        #loss(x_0, x_1, t, ps, st, rng, dev) = get_loss(
        #    x_0, x_1, t, drift, interpolant, gamma, ps, st, rng, dev
        #)
    end



    ode_sample(x_0, ps, st, dev) = StochasticInterpolants.ode_sampler(
        x_0, velocity, ps, st, num_steps, dev
    )

    return ConditionalStochasticInterpolant(
        velocity, score, sde_sample, ode_sample, interpolant, loss, gamma
    )
end
