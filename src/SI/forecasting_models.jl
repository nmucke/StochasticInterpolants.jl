function get_loss_function(;
    type::String,
    from_gaussian::Bool,
    velocity,
    interpolant
)

    if type == "forecasting"
        if from_gaussian
            return (x_condition, x_1, pars, ps, st, rng, dev) -> get_forecasting_from_gaussian_loss(
                x_condition, x_1, pars, velocity, interpolant, ps, st, rng, dev
            )
        else
            return (x_0, x_1, pars, ps, st, rng, dev) -> get_forecasting_loss(
                x_0, x_1, pars, velocity, interpolant, ps, st, rng, dev
            )
        end

    elseif type == "physics"
        return get_physics_forecasting_loss
    elseif type == "encoder"
        return get_encoder_forecasting_loss
    end
end





"""
    FollmerStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
struct FollmerStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:velocity, )
}
    velocity::Lux.AbstractExplicitLayer
    score
    interpolant
    loss
    gamma
    diffusion_coefficient
    drift_term
    diffusion_term
    projection
    len_history
    gaussian_base_distribution
end

"""
    FollmerStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer; 
        interpolant=Interpolant(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
    )
    
Constructs a Stochastic Interpolant model
"""
function FollmerStochasticInterpolant(
    velocity::Lux.AbstractExplicitLayer; 
    interpolant::NamedTuple,
    diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
    projection=nothing,
    len_history=1,
    dev=gpu_device(),
    gaussian_base_distribution=false
)

    gamma(t) = interpolant.gamma(t) |> Float32
    dgamma_dt(t) = interpolant.dgamma_dt(t) |> Float32

    alpha(t) = interpolant.alpha(t) |> Float32
    dalpha_dt(t) = interpolant.dalpha_dt(t) |> Float32

    beta(t) = interpolant.beta(t) |> Float32
    dbeta_dt(t) = interpolant.dbeta_dt(t) |> Float32
    
    diffusion_term(t, x, x_0, pars, ps, st) = begin
        return diffusion_coefficient(t)
    end

    score(input, vel_t) = begin

        x, x_0, t = input
        
        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;
        
        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0;
        
        return A .* (beta(t) .* vel_t .- c)
    end

    drift_term(t, x, x_0, pars, ps, st; ode_mode=false) = begin

        vel_t, st = velocity((x, x_0, pars, t), ps, st)

        if ode_mode
            return vel_t , st
            # return -2f0 .* x_0[:, :, :, end, :], st
        end

        return vel_t, st

        # out = (1f0 .+ 1f0./(2f0 .- t)) .* vel_t - 1f0./(t.*(2f0 .- t)) .* (2f0 .* x .- (2f0 .- t) .* x_0[:, :, :, end, :])

        # return out, st

        # A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        # A = 1 ./ A;
        # c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0[:, :, :, end, :];
        # score = A .* (beta(t) .* vel_t .- c)
        # return vel_t .+ 0.5f0 .* (diffusion_coefficient(t).^2 .- gamma(t).^2) .* score, st
    end

    # Loss including the score network
    loss = get_loss_function(
        type="forecasting",
        from_gaussian=gaussian_base_distribution,
        velocity=velocity,
        interpolant=interpolant
    )

    return FollmerStochasticInterpolant(
        velocity, score, interpolant, loss, gamma, diffusion_coefficient, drift_term, diffusion_term, projection, len_history, gaussian_base_distribution
    )
end


"""
    LatentFollmerStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
struct LatentFollmerStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:velocity, )
}
    velocity::Lux.AbstractExplicitLayer
    score::Function
    autoencoder::VAE_wrapper
    interpolant::NamedTuple
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
    projection
    len_history
end

"""
    LatentFollmerStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer,
        autoencoder::Lux.AbstractExplicitLayer;
        interpolant=Interpolant(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
    )
    
Constructs a Stochastic Interpolant model
"""
function LatentFollmerStochasticInterpolant(
    velocity::Lux.AbstractExplicitLayer,
    autoencoder::VAE_wrapper;
    interpolant::NamedTuple,
    diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
    projection=nothing,
    len_history=1,
    dev=gpu_device(),
    gaussian_base_distribution=false
)

    gamma(t) = interpolant.gamma(t)
    dgamma_dt(t) = interpolant.dgamma_dt(t)

    # _diffusion_coefficient(t) = diffusion_coefficient(t) #sqrt.((3f0 .- t) .* (1f0 .- t));#diffusion_coefficient.diffusion_coefficient(t)

    alpha(t) = interpolant.alpha(t)
    dalpha_dt(t) = interpolant.dalpha_dt(t)

    beta(t) = interpolant.beta(t)
    dbeta_dt(t) = interpolant.dbeta_dt(t)

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
            return vel_t , st
            # return -2f0 .* x_0, st
        end

        return vel_t, st

        # out = (1f0 .+ 1f0./(2f0 .- t)) .* vel_t - 1f0./(t.*(2f0 .- t)) .* (2f0 .* x .- (2f0 .- t) .* x_0)

        # return out, st

        # A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        # A = 1 ./ A;
        # c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0[:, :, :, end, :];
        # score = A .* (beta(t) .* vel_t .- c)
        # return vel_t .+ 0.5f0 .* (diffusion_coefficient(t).^2 .- gamma(t).^2) .* score, st
    end

    loss = get_loss_function(
        type="forecasting",
        from_gaussian=gaussian_base_distribution,
        velocity=velocity,
        interpolant=interpolant
    )

    return LatentFollmerStochasticInterpolant(
        velocity, score, autoencoder, interpolant, loss, gamma, diffusion_coefficient, drift_term, diffusion_term, projection, len_history
    )
end








"""
    EncoderFollmerStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
struct EncoderFollmerStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:encoder, :velocity, )
}
    velocity::Lux.AbstractExplicitLayer
    encoder::Lux.AbstractExplicitLayer
    score
    interpolant
    loss
    gamma
    diffusion_coefficient
    drift_term
    diffusion_term
    projection
end

"""
    EncoderFollmerStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer; 
        interpolant=Interpolant(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
    )
    
Constructs a Stochastic Interpolant model
"""
function EncoderFollmerStochasticInterpolant(
    velocity::Lux.AbstractExplicitLayer, 
    encoder::Lux.AbstractExplicitLayer,
    interpolant::NamedTuple;
    diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
    projection=nothing,
    dev=gpu_device()
)

    gamma(t) = interpolant.gamma(t)
    dgamma_dt(t) = interpolant.dgamma_dt(t)

    alpha(t) = interpolant.alpha(t)
    dalpha_dt(t) = interpolant.dalpha_dt(t)

    beta(t) = interpolant.beta(t)
    dbeta_dt(t) = interpolant.dbeta_dt(t)

    diffusion_term(t, x, x_0, pars, ps, st) = begin
        return diffusion_coefficient(t)
    end

    score(input, vel_t) = begin

        x, x_0, t = input
        
        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;
        
        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0;
        
        return A .* (beta(t) .* vel_t .- c)
    end

    drift_term(t, x, x_0, pars, ps, st; ode_mode=false) = begin

        vel_t, st = velocity((x, x_0, pars, t), ps, st)

        if ode_mode
            # return vel_t , st
            return -2f0 .* x_0[:, :, :, end, :], st
        end

        out = (1f0 .+ 1f0./(2f0 .- t)) .* vel_t - 1f0./(t.*(2f0 .- t)) .* (2f0 .* x .- (2f0 .- t) .* x_0[:, :, :, end, :])

        return out, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_encoder_forecasting_loss(
        x_0, x_1, pars, velocity, encoder, interpolant, ps, st, rng, dev
    )

    return EncoderFollmerStochasticInterpolant(
        velocity, encoder, score, interpolant, loss, gamma, diffusion_coefficient, drift_term, diffusion_term, projection
    )
end
































"""
    DataDependentCouplingStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer
        score::Lux.AbstractExplicitLayer

    )
    

A container layer for the Stochastic Interpolant model
"""
struct DataDependentCouplingStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:velocity, :score)
}
    velocity::Lux.AbstractExplicitLayer
    score::Lux.AbstractExplicitLayer
    interpolant::NamedTuple
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
    projection
end

"""
    DataDependentCouplingStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer; 
        interpolant=Interpolant(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
    )
    
Constructs a Stochastic Interpolant model
"""
function DataDependentCouplingStochasticInterpolant(
    velocity::Lux.AbstractExplicitLayer,
    score::Lux.AbstractExplicitLayer;
    interpolant=Interpolant(),
    diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
    projection=nothing,
    dev=gpu_device()
)

    gamma(t) = interpolant.gamma(t)
    dgamma_dt(t) = interpolant.dgamma_dt(t)

    alpha(t) = interpolant.alpha(t)
    dalpha_dt(t) = interpolant.dalpha_dt(t)

    beta(t) = interpolant.beta(t)
    dbeta_dt(t) = interpolant.dbeta_dt(t)

    diffusion_term(x, ps, st) = begin
        x, x_0, pars, t = x
        return diffusion_coefficient(t)
    end


    drift_term(x, ps, st; ode_mode=false) = begin
        
        x, x_0, pars, t = x

        vel_t, st_new = velocity((x, x_0, pars, t), ps.velocity, st.velocity)
        @set st.velocity = st_new

        if ode_mode
            return vel_t, st
        end

        score_term, st_new = score((x, x_0, pars, t), ps.score, st.score)
        @set st.score = st_new

        score_multiplication_term = 0.5f0 .* sqrt.(diffusion_coefficient(t)) ./ gamma(t)
        score_term = score_multiplication_term .* score_term

        return vel_t .- score_term, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_forecasting_loss(
        x_0, x_1, pars, velocity, score, interpolant, ps, st, rng, dev
    )

    return DataDependentCouplingStochasticInterpolant(
        velocity, score, interpolant, loss, gamma, diffusion_coefficient, drift_term, diffusion_term, projection
    )
end

"""
DataDependentCouplingStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
# struct DataDependentCouplingStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
#     (:g_0, :g_1, :g_z)
# }
#     g_0::Lux.AbstractExplicitLayer; 
#     g_1::Lux.AbstractExplicitLayer; 
#     g_z::Lux.AbstractExplicitLayer; 
#     velocity::Function;
#     interpolant::Function
#     loss::Function
#     gamma::Function
#     diffusion_coefficient::Function
#     drift_term::Function
#     diffusion_term::Function
# end

"""
DataDependentCouplingStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer; 
        interpolant=Interpolant(),
        gamma=Gamma(),
        diffusion_coefficient=DiffusionCoefficient(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
    interpolant::Interpolant
    )
    
Constructs a Stochastic Interpolant model
"""
function DataDependentCouplingStochasticInterpolant(
    g_0::Lux.AbstractExplicitLayer,
    g_1::Lux.AbstractExplicitLayer, 
    g_z::Lux.AbstractExplicitLayer; 
    interpolant=Interpolant(),
    gamma=Gamma(),
    diffusion_coefficient=DiffusionCoefficient(),
    diffusion_multiplier=0.1f0,
    dev=gpu_device()
)

    _gamma(t) = diffusion_multiplier .* gamma.gamma(t)
    dgamma_dt(t) = diffusion_multiplier .* gamma.dgamma_dt(t)

    _diffusion_coefficient(t) = diffusion_multiplier .* diffusion_coefficient.diffusion_coefficient(t)

    alpha(t) = interpolant.alpha(t)
    dalpha_dt(t) = interpolant.dalpha_dt(t)

    beta(t) = interpolant.beta(t)
    dbeta_dt(t) = interpolant.dbeta_dt(t)

    diffusion_term(t, x, pars, ps, st) = begin
        return _diffusion_coefficient(t)
    end

    score(t, x, x_0, pars, ps, st) = begin

        _g_z, st_new  = g_z((x, x_0, pars, t), ps.g_z, st.g_z)
        @set st.g_z = st_new

        return _g_z ./ gamma(t), st

    end

    velocity(input, ps, st) = begin

        x, x_0, pars, t = input

        _g_0, st_new = g_0((x, x_0, pars, t), ps.g_0, st.g_0)
        @set st.g_0 = st_new

        _g_1, st_new  = g_1((x, x_0, pars, t), ps.g_1, st.g_1)
        @set st.g_1 = st_new

        _g_z, st_new  = g_z((x, x_0, pars, t), ps.g_z, st.g_z)
        @set st.g_z = st_new

        _velocity = dalpha_dt(t) .* _g_0 .+ dbeta_dt(t) .* _g_1 .+ gamma(t) .* _g_z, st

        return _velocity, st
    end

    drift_term(t, x, x_0, pars, ps, st; ode_mode=false) = begin

        _velocity, st = velocity((x, x_0, pars, t), ps, st)

        if ode_mode
            return _velocity, st
        end
        
        _score, st = score(t, x, x_0, pars, ps, st) 

        _diffusion_term = diffusion_term(t, x, pars, ps, st)
        _diffusion_term = 0.5f0 .* diffusion_term .^ 2

        return _velocity .- _diffusion_term .* _score, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_forecasting_loss(
        x_0, x_1, pars, g_0, g_1, g_z, interpolant, _gamma, ps, st, rng, dev
    )

    return ForecastingStochasticInterpolant(
        g_0, g_1, g_z, interpolant, loss, _gamma, _diffusion_coefficient, drift_term, diffusion_term
    )
end


"""
    FollmerStochasticInterpolant(
    )
    

A container layer for the Stochastic Interpolant model
"""
struct EntropicActionMatching <: Lux.AbstractExplicitContainerLayer{
    (:action, :velocity)
}
    action::Lux.AbstractExplicitLayer
    velocity::Function
    loss::Function
    interpolant::NamedTuple
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
    projection
end

"""
    EntropicActionMatching(
        velocity::Lux.AbstractExplicitLayer; 
        diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
        projection=nothing,
        dev=gpu_device()
    )
    
Constructs a Stochastic Interpolant model
"""
function EntropicActionMatching(
    action::Lux.AbstractExplicitLayer;
    interpolant=Interpolant(),
    diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
    projection=nothing,
    dev=gpu_device()
)

    diffusion_term(t, x, x_0, pars, ps, st) = begin
        return diffusion_coefficient(t)
    end

    drift_term(t, x, x_0, pars, ps, st; ode_mode=false) = begin

        vel_t, st = velocity((x, x_0, pars, t), ps, st)

        return vel_t, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_action_matching_loss(
        x_0, x_1, pars, action, interpolant, diffusion_coefficient, ps, st, rng, dev
    )

    return EntropicActionMatching(
        action, velocity, loss, diffusion_coefficient, drift_term, diffusion_term, projection
    )
end
