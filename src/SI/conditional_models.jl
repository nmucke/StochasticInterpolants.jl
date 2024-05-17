

"""
ConditionalStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    ) where T <: AbstractFloat
    

A container layer for the Stochastic Interpolant model
"""
struct ConditionalStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:unet, )
}
    unet::UNet
    sde_sample::Function
    ode_sample::Function
    interpolant::Function
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
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0,
    max_freq=1000.0f0,
    embedding_dims=32,
    num_steps=100,
)

    
    unet = UNet(
        image_size; 
        in_channels=in_channels,
        channels=channels, 
        block_depth=block_depth,
        min_freq=min_freq, 
        max_freq=max_freq, 
        embedding_dims=embedding_dims,
    )

    interpolant = StochasticInterpolants.linear_interpolant


    sde_sample(num_samples, ps, st, rng, dev) = StochasticInterpolants.sde_sampler(
        unet,
        ps,
        st,
        rng,
        marginal_probability_std,
        diffusion_coefficient,
        num_samples,
        num_steps,
        eps,
        dev
    )

    ode_sample(num_samples, ps, st, rng, x, dev) = StochasticInterpolants.ode_sampler(
        unet,
        ps,
        st,
        rng,
        num_samples,
        num_steps,
        x,
        dev
    )
    
    return ConditionalStochasticInterpolant(
        unet, sde_sample, ode_sample, interpolant
    )
end