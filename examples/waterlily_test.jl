


JULIA_NUM_THREADS=auto
using CUDA
using WaterLily

using Interpolations
using StochasticInterpolants

using Adapt
# Choose between "transonic_cylinder_flow", "incompressible_flow", "turbulence_in_periodic_box"
test_case = "incompressible_flow";

# Which type of testing to perform
# options are "pars_extrapolation", "pars_interpolation", "long_rollouts" for "transonic_cylinder_flow" test case
# options are "pars_low", "pars_high", "pars_var" for "incompressible_flow" test case
test_args = "pars_low";

trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars = load_test_case_data(
    test_case, 
    test_args,
);


using Plots


t = 20
lol = projection_with_obstacle(trainset[:,:,:,t,:], CuArray);



function get_interpolation(data, dev)
    x_grid = 0:size(data, 1)-1
    y_grid = 0:size(data, 2)-1

    u_interp_x = linear_interpolation((x_grid, y_grid), data[:, :, 1]; extrapolation_bc = Line())
    u_interp_y = linear_interpolation((x_grid, y_grid), data[:, :, 2]; extrapolation_bc = Line())

    if dev == CuArray
        cu_u_interp_x = adapt(CuArray{Float32}, u_interp_x)
        cu_u_interp_y = adapt(CuArray{Float32}, u_interp_y)
    else
        cu_u_interp_x = u_interp_x
        cu_u_interp_y = u_interp_y
    end

    fun = (i, x) -> begin
        x, y = x
        i==1 && return cu_u_interp_x(x, y)
        i==2 && return cu_u_interp_y(x, y)
        return 0 
    end
    return fun
end

function waterlily_projection(u, dev)

    n, m = size(u, 2), size(u, 1)

    u_interp = get_interpolation(permutedims(u, (2,1,3)), dev)

    # signed distance function to circle
    radius, center = 0.3/2*m, 1/4*n
    sdf(x,t) = √sum(abs2, x .- center) - radius

    U = 0.5
    Re = 100
    circ = Simulation(
        (n,m),   # domain size
        (U, 0.0),   # domain velocity (& velocity scale)
        2*radius; # length scale
        ν=U*2*radius/Re,     # fluid viscosity
        body=AutoBody(sdf), # geometry
        Δt=0.1,
        U=U,
        uλ=u_interp,
        exitBC=true,
        perdir=(1,2),
        T=Float32,
        mem=dev
    ) 

    WaterLily.exitBC!(circ.flow.u,circ.flow.u⁰,U,circ.flow.Δt[end])
    WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)
    WaterLily.project!(circ.flow, circ.pois)
    WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)

    # proj_interp = get_interpolation(circ.flow.u, dev)

    x_grid = 0:size(circ.flow.u, 1)-1
    y_grid = 0:size(circ.flow.u, 2)-1
    proj_interp_x = linear_interpolation((x_grid, y_grid), circ.flow.u[:, :, 1]; extrapolation_bc = Line())
    proj_interp_y = linear_interpolation((x_grid, y_grid), circ.flow.u[:, :, 2]; extrapolation_bc = Line())

    if dev == CuArray
        proj_interp_x = adapt(CuArray{Float32}, proj_interp_x)
        proj_interp_y = adapt(CuArray{Float32}, proj_interp_y)
        u_proj_x = proj_interp_x(1:n, 1:m)
        u_proj_y = proj_interp_y(1:n, 1:m)
    else
        u_proj_x = proj_interp_x(1:n, 1:m)
        u_proj_y = proj_interp_y(1:n, 1:m)
    end

    u_proj = cat(u_proj_x, u_proj_y, dims=3)

    return permutedims(u_proj, (2,1,3))
end

t = 20

@time lol = waterlily_projection(testset[:,:,:,t,1], Array);

sum(abs2, lol[:,:,1] .* mask - testset[:,:,1,t,1] .* mask) / sum(abs2, testset[:,:,1,t,1] .* mask)

p1 = heatmap(lol[:, :, 1] .* mask, title="Projected")
p2 = heatmap(testset[:, :, 1, t, 1] .* mask, title="Original")
e = lol[:, :, 1] .* mask - testset[:, :, 1, t, 1] .* mask
p3 = heatmap(e, title="Difference")
plot(p1, p2, p3, layout=(2,2), size=(1200,1200))

heatmap(lol[4:end-1, 4:end-1, 1])
heatmap(testset[1:end-4, 1:end-4, 1, t, 1])


u = testset[:,:,:,t,1] + testset[:,:,:,t,2]
n, m = size(u, 2), size(u, 1)


u_interp = get_interpolation(permutedims(u |> CuArray, (2,1,3)))

lol = u_interp(1, (1:n, 1:m))

# signed distance function to circle
radius, center = 0.3/2*m, 1/4*n
WaterLily.sdf(x,t) = √sum(abs2, x .- center) - radius

body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
U = 0.5
Re = 100
mem = CuArray


function circle(n,m;Re=250,U=0.5,mem=Array)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation(
        (n,m),   # domain size
        (U,0.0),   # domain velocity (& velocity scale)
        2*radius; # length scale
        ν=U*2*radius/Re,     # fluid viscosity
        body=body, # geometry
        Δt=0.1,
        uλ=u_interp,
        mem=mem
    ) 
end

circ = circle(128,64,mem=mem)



lol = circ.flow.u

WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)
WaterLily.project!(circ.flow, circ.pois,0.5);
WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)

lol-circ.flow.u

function circle(n,m;Re=250,U=1,mem=Array)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation(
        (n,m),   # domain size
        (U,0),   # domain velocity (& velocity scale)
        2*radius; # length scale
        ν=U*2*radius/Re,     # fluid viscosity
        body=body, # geometry
        Δt=0.1,
        mem=mem
    ) 
end

# Initialize the simulation with GPU Array
using CUDA
sim = circle(3*2^6,2^7; mem=CuArray);


# WaterLily.BDIM!(circ.flow); 
WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)
WaterLily.project!(circ.flow, circ.pois)
WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)

proj_interp = get_interpolation(circ.flow.u)
u_proj_x = proj_interp(1, (1:n, 1:m))
u_proj_y = proj_interp(2, (1:n, 1:m))

u_proj = cat(u_proj_x, u_proj_y, dims=3)

























u_interp = get_interpolation(permutedims(u, (2,1,3)))

# signed distance function to circle
radius, center = 0.3/2*m, 1/4*n
WaterLily.sdf(x,t) = √sum(abs2, x .- center) - radius

U = 0.5
Re = 100
circ = Simulation(
    (n,m),   # domain size
    (U,0),   # domain velocity (& velocity scale)
    2*radius; # length scale
    ν=U*2*radius/Re,     # fluid viscosity
    body=AutoBody(sdf), # geometry
    Δt=0.1,
    U=U,
    uλ=u_interp,
    exitBC=true,
)#mem=CuArray) # memory location
lol = circ.flow.u

WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)
WaterLily.project!(circ.flow, circ.pois,0.5);
WaterLily.BC!(circ.flow.u,U,circ.flow.exitBC,circ.flow.perdir)

lol-circ.flow.u



function circle(n,m;Re=100,U=0.5*32)
    # signed distance function to circle
    radius, center = 0.3/2*m, n/4    #m/8, m/2-1
    sdf(x,t) = √sum(abs2, x .- center) - radius

    Simulation((n,m),   # domain size
               (U,0),   # domain velocity (& velocity scale)
               2*radius; # length scale
               ν=U*2*radius/Re,     # fluid viscosity
               body=AutoBody(sdf), # geometry
               Δt=0.1,
               uλ=get_interpolation((128, 64), permutedims(testset[:,:,:,1,1], (2,1,3))),
               exitBC=true,
               )#mem=CuArray) # memory location
end


n = 128
m = 64
circ = circle(n,m)

out = zeros(64+2,128+2,2,600)
for i in 1:6000
    sim_step!(circ)
    if i%10 == 0
        out[:,:,1,Int(i/10)] = circ.flow.u[:,:,1]'
        out[:,:,2,Int(i/10)] = circ.flow.u[:,:,2]'
    end
end

plot_out = sqrt.(out[:, :, 1, :].^2 + out[:, :, 2, :].^2)

create_gif([plot_out], "waterlily_test.gif", ["lol"])










uu_padded[:, :, 2] - u_lambda(2, (x_grid_padded, y_grid_padded))


ff = WaterLily.Flow(
    (n,m), (1, 0); 
    uλ = u_interp,
    T=Float32
)#, f=CuArray)


a = 2*0.3/100


b = 2*0.3/2*64/100

b/a

Ng = (64, 128) .+ 2
Nd = (Ng..., D)
u = Array{T}(undef, Nd...)


ff = WaterLily.Flow(
    (n,m), (0.5, 0); 
    uλ = u_lambda,
    Δt=0.05,
    T=Float32
)#, f=CuArray)



circ.flow.u[2:end-1,2:end-1,1]' - testset[:,:,1,1,1]


plot_out = sqrt.(uuu[:,:,1].^2 + uuu[:,:,2].^2)
test_out = sqrt.(sum(testset.^2, dims=3))[:, :, 1, :, 1]







lol1 = u_lambda(1, (x_grid_padded,y_grid_padded ))
lol2 = u_lambda(2, (x_grid_padded,y_grid_padded ))

WaterLily.project!(ff, circ.pois)


lol1 = circ.flow.u[:,:,1] |> Array
lol2 = circ.flow.u[:,:,2] |> Array

lol = sqrt.(lol1.^2 + lol2.^2)
test_out = sqrt.(testset[:, :, 1, 10, 1].^2 + testset[:, :, 2, 10, 1].^2)

heatmap(test_out[:,:,1])
heatmap(lol[2:end-1,2:end-1,1]')
heatmap(plot_out[:,:,10]')


test_out[:,:,1] - lol[2:end-1,2:end-1,1]'

testset[:,:,1,1,1] - lol1[2:end-1,2:end-1,1]'






plot_out[2:end-1,2:end-1,1]' - test_out[:,:,1]




using Plots
u = circ.flow.u[:,:,1] |> Array # first component is x
contourf(u') # transpose the array for the plot

























uu = circ.flow.u |> Array
# uu = uu[2:end-1,2:end-1,1]

x = range(-2, size(uu,2), length=size(uu,2))
y = range(-2, size(uu,1), length=size(uu,1))

u_interp_x = interpolate((y, x), uu[:, :, 1], Gridded(Linear()))
u_interp_y = interpolate((y, x), uu[:, :, 2], Gridded(Linear()))

interp = (x,y) -> [u_interp_x(x,y), u_interp_y(x,y)]

ff = WaterLily.Flow(
    (n,m), (1, 0); 
    uλ = (i,x) -> begin
        println(x)
        i==1 && return u_interp_x(x[1], x[2])
        i==2 && return u_interp_y(x[1], x[2])
        return 0
    end,
    T=Float32
)#, f=CuArray)

WaterLily.project!(ff, circ.pois)
u = ff.u[:,:,1] |> Array
contourf(u')





using WaterLily
using Plots; gr()
using StaticArrays

# required to keep things global
let
    # parameters
    Re = 250; U = 1
    p = 5; L = 2^p
    radius, center = L/2, 2L

    # fsi parameters
    T = 4*radius    # VIV period
    mₐ = π*radius^2 # added-mass coefficent circle
    m = 0.1*mₐ      # mass as a fraction of the added-mass, can be zero
    k = (2*pi/T)^2*(m+mₐ)

    # initial condition FSI
    p0=radius/3; v0=0; a0=0; t0=0

    # motion function uses global var to adjust
    posx(t) = p0 + (t-t0)*v0

    # motion definition
    map(x,t) = x - SA[0, posx(t)]

    # mₐke a body
    circle = AutoBody((x,t)->√sum(abs2, x .- center) - radius, map)

    # generate sim
    sim = Simulation((6L,4L), (U,0), radius; ν=U*radius/Re, body=circle)

    # get start time
    duration=10; step=0.1; t₀=round(sim_time(sim))

    @time @gif for tᵢ in range(t₀,t₀+duration;step)

        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U

            # measure body
            measure!(sim,t)

            # update flow
            mom_step!(sim.flow,sim.pois)

            # pressure force
            force = -WaterLily.pressure_force(sim)

            # compute motion and acceleration 1DOF
            Δt = sim.flow.Δt[end]
            accel = (force[2]- k*p0 + mₐ*a0)/(m + mₐ)
            p0 += Δt*(v0+Δt*accel/2.)
            v0 += Δt*accel
            a0 = accel

            # update time, sets the pos/v0 correctly
            t0 = t; t += Δt
        end

        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ; shift=(-0.5,-0.5),clims=(-5,5))
        body_plot!(sim); plot!(title="tU/L $tᵢ")

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end


















u = circ.flow.u

WaterLily.div(circ.flow)



using WaterLily
using Plots

cID = "2DCircle"

function circle(n,m;Re=250,U=1,mem=Array)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem)
end

# Initialize the simulation with GPU Array
using CUDA
sim = circle(3*2^6,2^7; mem=CuArray);

WaterLily.logger(cID) # Log the residual of pressure solver
#= NOTE: 
If you want to log residuals during a GPU simulation, it's better to include the following line. 
Otherwise, Julia will generate excessive debugging messages, which can significantly slow down the simulation. 
=#
using Logging; disable_logging(Logging.Debug)

circ = circle(3*2^5,2^6)
for i in 1:1000
    sim_step!(circ)
end

# Run the simulation
sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true)

# Remember to call Plots package (already done in Line 2). This will let WaterLily
# knows you want to plot sth like residual and will compile the funciton for you.
# NOTE: Comment out this line if you want to see gif animation!
plot_logger("$(cID).log")