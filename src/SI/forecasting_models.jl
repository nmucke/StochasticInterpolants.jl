
"""
    ForecastingStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
struct ForecastingStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:velocity, )
}
    velocity::Lux.AbstractExplicitLayer
    score::Function
    interpolant::Function
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
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
    diffusion_multiplier=0.1f0,
    dev=gpu_device()
)

    diffusion_multiplier = diffusion_multiplier |> dev

    gamma(t) = diffusion_multiplier .* (1f0 .- t) # |> dev
    # dgamma_dt(t) = -diffusion_multiplier .* ones(size(t))  # |> dev
    dgamma_dt(t) = -diffusion_multiplier * ones(size(t)) |> dev #fill!(similar(t, size(t)), -diffusion_multiplier)

    diffusion_coefficient(t) = diffusion_multiplier .* sqrt.((3f0 .- t) .* (1f0 .- t)) # |> dev
    # diffusion_coefficient(t) = diffusion_multiplier .* gamma(t) |> dev

    alpha(t) = 1f0 .- t # |> dev
    dalpha_dt(t) = -1f0  # |> dev

    beta(t) = t.^2 # |> dev
    dbeta_dt(t) = 2f0 .* t # |> dev

    interpolant(x_0, x_1, t) = begin

        I = alpha(t) .* x_0 .+ beta(t) .* x_1
        dI_dt =  dalpha_dt(t) .* x_0 + dbeta_dt(t) .* x_1

        return I, dI_dt
    end


    # velocity = ConditionalUNet(
    #     image_size; 
    #     in_channels=in_channels,
    #     channels=channels, 
    #     block_depth=block_depth,
    #     min_freq=min_freq, 
    #     max_freq=max_freq, 
    #     embedding_dims=embedding_dims,
    # )

    # velocity =  ConvNextUNet(
    #     image_size; 
    #     in_channels=in_channels,
    #     channels=channels, 
    #     block_depth=block_depth,
    #     min_freq=min_freq, 
    #     max_freq=max_freq, 
    #     embedding_dims=embedding_dims
    # )
    # velocity = DitParsConvNextUNet(
    #     image_size; 
    #     in_channels=in_channels,
    #     channels=channels, 
    #     block_depth=block_depth,
    #     min_freq=min_freq, 
    #     max_freq=max_freq, 
    #     embedding_dims=embedding_dims,
    #     pars_dim=1
    # )
    velocity = AttnParsConvNextUNet(
        image_size; 
        in_channels=in_channels,
        channels=channels, 
        block_depth=block_depth,
        min_freq=min_freq, 
        max_freq=max_freq, 
        embedding_dims=embedding_dims,
        pars_dim=1
    )

    # velocity = ConditionalDiffusionTransformer(
    #     image_size;
    #     in_channels=in_channels, 
    #     patch_size=(8, 8),
    #     embed_dim=256, 
    #     depth=4, 
    #     number_heads=8,
    #     mlp_ratio=4.0f0, 
    #     dropout_rate=0.1f0, 
    #     embedding_dropout_rate=0.1f0,
    # )

    diffusion_term(t, x, x_0, pars, ps, st) = begin
        return diffusion_coefficient(t)
    end


    score(input, vel_t) = begin

        x, x_0, t = input
        
        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;
        
        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0;
        
        # A = repeat(A, size(x_0)[1:3]...)
        # beta_ = repeat(beta(t), size(x_0)[1:3]...)
        
        return A .* (beta(t) .* vel_t .- c)
    end

    drift_term(t, x, x_0, pars, ps, st; ode_mode=false) = begin

        vel_t, st = velocity((x, x_0, pars, t), ps, st)

        if ode_mode

            # score = 0.5f0 .* score

            out = vel_t
            
            return out , st
        end

        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;

        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0;

        score = A .* (beta(t) .* vel_t .- c)


        return vel_t .+ 0.5f0 .* (diffusion_coefficient(t).^2 .- gamma(t).^2) .* score, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_forecasting_loss(
        x_0, x_1, pars, velocity, interpolant, gamma, dgamma_dt, ps, st, rng, dev
    )

    return ForecastingStochasticInterpolant(
        velocity, score, interpolant, loss, gamma, diffusion_coefficient, drift_term, diffusion_term
    )
end
