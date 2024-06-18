
"""
ForecastingStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    ) where T <: AbstractFloat
    

A container layer for the Stochastic Interpolant model
"""
struct ForecastingStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:velocity, )
}
    velocity::ConditionalUNet
    score::Function
    interpolant::Function
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
end

"""
ForecastingStochasticInterpolant(
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
function ForecastingStochasticInterpolant(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0,
    max_freq=1000.0f0,
    embedding_dims=32,
    num_steps=100,
    dev=gpu
)

    diff_multiplier = 0.1f0 |> dev

    gamma(t) = diff_multiplier .* (1 .- t) |> dev
    dgamma_dt(t) = -diff_multiplier .* ones(size(t))  |> dev

    diffusion_coefficient(t) = diff_multiplier .* sqrt.((3 .- t) .* (1 .- t))  |> dev

    alpha(t) = 1 .- t  |> dev
    dalpha_dt(t) = -1  |> dev

    beta(t) = t.^2  |> dev
    dbeta_dt(t) = 2 .* t  |> dev

    interpolant(x_0, x_1, t) = begin

        I = alpha(t) .* x_0 .+ beta(t) .* x_1
        dI_dt =  dalpha_dt(t) .* x_0 + dbeta_dt(t) .* x_1

        return I, dI_dt
    end

    velocity = ConditionalUNet(
        image_size; 
        in_channels=in_channels,
        channels=channels, 
        block_depth=block_depth,
        min_freq=min_freq, 
        max_freq=max_freq, 
        embedding_dims=embedding_dims,
    )

    score(input, vel_t) = begin

        x, x_0, t = input
        
        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;
        
        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0;
        
        # A = repeat(A, size(x_0)[1:3]...)
        # beta_ = repeat(beta(t), size(x_0)[1:3]...)
        
        return A .* (beta(t) .* vel_t .- c)
    end

    # Loss including the score network
    loss(x_0, x_1, ps, st, rng, dev) = get_forecasting_loss(
        x_0, x_1, velocity, interpolant, gamma, dgamma_dt, ps, st, rng, dev
    )

    return ForecastingStochasticInterpolant(
        velocity, score, interpolant, loss, gamma, diffusion_coefficient
    )
end
