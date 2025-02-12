using LinearAlgebra
using IncompressibleNavierStokes
using CUDA
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

function projection_with_obstacle(u, dev)

    u = u |> Array
    out = Array{eltype(u)}[]
    for i in 1:size(u, 4)
        u_proj = waterlily_projection(u[:, :, :, i], Array)
        out = [out; [u_proj]]
    end
    out = stack(out; dims = 4) |> dev
    return out

end


function divfunc(u, dev)

    ax_x = LinRange(0f0, 2f0*pi, 128 + 1) 
    ax_y = LinRange(0f0, 2f0*pi, 128 + 1)
    Re = 1000f0;
    setup = IncompressibleNavierStokes.Setup(; x = (ax_x, ax_y), Re, ArrayType = CuArray{Float32});
    create_divergence(setup, psolver) = function divergence(u)
        
        u = pad_circular(u, 1; dims = 1:2)
        
        out = Array{eltype(u)}[]
        for i in 1:size(u, 4)
            u_ = u[:, :, 1, i], u[:, :, 2, i]
            u_ = IncompressibleNavierStokes.divergence(u_, setup)
            u_ = IncompressibleNavierStokes.scalewithvolume(u_, setup)
            u_ = u_[2:end-1, 2:end-1, :]
            out = [out; [u_]]
        end
        stack(out; dims = 4)
    end

    div = create_divergence(setup, nothing);

    return div(u |> dev)
    
end

function gradfunc(q, dx)
    q_grad_x = (circshift(q, (0, 0)) - circshift(q, (1, 0))) / dx;
    q_grad_y = (circshift(q, (0, 0)) - circshift(q, (0, 1))) / dx;
    q_grad_x, q_grad_y
end

function laplacefunc(q, dx)
    q_laplace = circshift(q, (-1, 0)) + circshift(q, (1, 0)) + circshift(q, (0, -1)) + circshift(q, (0, 1)) - 4q;
    q_laplace / dx^2
end


function project_onto_divergence_free(u, dev)

    ax_x = LinRange(0f0, 2f0*pi, 128 + 1) 
    ax_y = LinRange(0f0, 2f0*pi, 128 + 1)
    Re = 1000f0;
    setup = IncompressibleNavierStokes.Setup(; x = (ax_x, ax_y), Re, ArrayType = CuArray{Float32});
    create_projection(setup, psolver) = function projection(u)
        
        u = pad_circular(u, 1; dims = 1:2)
        
        out = Array{eltype(u)}[]
        for i in 1:size(u, 4)
            u_ = u[:, :, 1, i], u[:, :, 2, i]
            u_ = IncompressibleNavierStokes.project(u_, setup; psolver)
            u_ = cat(u_[1], u_[2]; dims = 3)
            u_ = u_[2:end-1, 2:end-1, :]
            out = [out; [u_]]
        end
        stack(out; dims = 4)
    end

    psolver = IncompressibleNavierStokes.psolver_spectral(setup)
    p = create_projection(setup, psolver);

    return p(u |> dev)
end


# function project_onto_divergence_free(u, dev)

#     dx = 1/(size(u, 1)-1) |> dev;

#     batch_size = size(u, 4);

#     u_x = u[:, :, 1, :];
#     u_y = u[:, :, 2, :];

#     div = divfunc(u_x, u_y, dx);

#     div_RHS = reshape(div, size(div)[1]*size(div)[2], batch_size) |> cpu_device();

#     possion_matrix_1D = Tridiagonal(
#         ones(size(div, 1)-1),
#         -4*ones(size(div, 1)),
#         ones(size(div, 1)-1)
#     );
#     possion_matrix_1D = Matrix(possion_matrix_1D);
#     possion_matrix_1D[1, end] = 1;
#     possion_matrix_1D[end, 1] = 1;

#     S = Tridiagonal(
#         ones(size(div, 1)-1),
#         zeros(size(div, 1)),
#         ones(size(div, 1)-1)
#     );
#     S = Matrix(S);
#     S[1, end] = 1;
#     S[end, 1] = 1;

#     possion_matrix_2D = kron(I(size(div, 2)), possion_matrix_1D) + kron(S, I(size(div, 2)));
#     possion_matrix_2D = possion_matrix_2D / (dx * dx);
#     possion_matrix_2D = possion_matrix_2D;

#     q = zeros(size(possion_matrix_2D, 1), batch_size);
#     for i = 1:batch_size
#         q[:, i] = possion_matrix_2D \ div_RHS[:, i];
#     end
#     q = reshape(q, size(div));

#     q_grad_x, q_grad_y = gradfunc(q, dx) .|> dev;

#     u_x_proj = u_x - q_grad_x;
#     u_y_proj = u_y - q_grad_y;

#     u_x_proj = reshape(u_x_proj, size(u_x)..., 1);
#     u_y_proj = reshape(u_y_proj, size(u_y)..., 1);

#     u_x_proj = permutedims(u_x_proj, [1, 2, 4, 3]);
#     u_y_proj = permutedims(u_y_proj, [1, 2, 4, 3]);

#     return cat(u_x_proj, u_y_proj, dims=3)
# end

