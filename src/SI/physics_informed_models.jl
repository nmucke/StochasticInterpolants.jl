
"""
    PhysicsInformedStochasticInterpolant(
        unet::UNet,
        sde_sample::Function,
        ode_sample::Function,
        interpolant::Function
    )
    

A container layer for the Stochastic Interpolant model
"""
struct PhysicsInformedStochasticInterpolant <: Lux.AbstractExplicitContainerLayer{
    (:model_velocity, )
}
    velocity::Function
    model_velocity::Lux.AbstractExplicitLayer
    physics_velocity::Function
    score::Function
    interpolant::NamedTuple
    loss::Function
    gamma::Function
    diffusion_coefficient::Function
    drift_term::Function
    diffusion_term::Function
    projection
end

"""
    PhysicsInformedStochasticInterpolant(
        velocity::Lux.AbstractExplicitLayer; 
        interpolant=Interpolant(),
        diffusion_multiplier=0.1f0,
        dev=gpu_device() 
    )
    
Constructs a Stochastic Interpolant model
"""
function PhysicsInformedStochasticInterpolant(
    model_velocity::Lux.AbstractExplicitLayer,
    physics_velocity::Function; 
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

    velocity(input, ps, st) = begin

        model_vel, st = model_velocity(input, ps, st)

        return model_vel + physics_velocity(input), st
    end

    score(input, vel_t) = begin

        x, x_0, pars, t = input
        
        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;
        
        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0;
        
        # A = repeat(A, size(x_0)[1:3]...)
        # beta_ = repeat(beta(t), size(x_0)[1:3]...)
        
        return A .* (beta(t) .* vel_t .- c)
    end

    drift_term(t, x, x_0, pars, ps, st; ode_mode=false, phys_vel=nothing) = begin

        model_vel_t, st = model_velocity((x, x_0, pars, t), ps, st)

        if !isnothing(phys_vel)
            vel_t = model_vel_t #.+ phys_vel
        else
            vel_t = model_vel_t #.+ physics_velocity((x, x_0, pars, t))
        end

        if ode_mode
            return vel_t , st
        end

        A = t .* gamma(t) .* (dbeta_dt(t) .* gamma(t) .- beta(t) .* dgamma_dt(t));
        A = 1 ./ A;

        c = dbeta_dt(t) .* x .+ (beta(t) .* dalpha_dt(t) - alpha(t) .* dbeta_dt(t)) .* x_0[:, :, :, end, :];

        score = A .* (beta(t) .* vel_t .- c)

        return vel_t .+ 0.5f0 .* (diffusion_coefficient(t).^2 .- gamma(t).^2) .* score, st
    end

    # Loss including the score network
    loss(x_0, x_1, pars, ps, st, rng, dev) = get_physics_forecasting_loss(
        x_0, x_1, pars, model_velocity, physics_velocity, interpolant, ps, st, rng, dev
    )

    return PhysicsInformedStochasticInterpolant(
        velocity, model_velocity, physics_velocity, score, interpolant, loss, gamma, diffusion_coefficient, drift_term, diffusion_term, projection
    )
end

