
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
    score::Function
    interpolant::Interpolant
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
    projection
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
    interpolant=Interpolant(),
    diffusion_coefficient=DiffusionCoefficient(t -> sqrt.((3f0 .- t) .* (1f0 .- t))),
    projection=nothing,
    dev=gpu_device()
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

            # score = 0.5f0 .* score

            out = vel_t
            
            return out , st
        end

        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;

        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0[:, :, :, end, :];

        score = A .* (beta(t) .* vel_t .- c)

        return vel_t .+ 0.5f0 .* (diffusion_coefficient(t).^2 .- gamma(t).^2) .* score, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_forecasting_loss(
        x_0, x_1, pars, velocity, interpolant, ps, st, rng, dev
    )

    return FollmerStochasticInterpolant(
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
struct DataDependentCouplingStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:g_0, :g_1, :g_z)
}
    g_0::Lux.AbstractExplicitLayer; 
    g_1::Lux.AbstractExplicitLayer; 
    g_z::Lux.AbstractExplicitLayer; 
    velocity::Function;
    interpolant::Function
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
end

"""
DataDependentCouplingStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer; 
        interpolant=Interpolant(),
        gamma=Gamma(),
        diffusion_coefficient=DiffusionCoefficient(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
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
            return _velocity
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
