
using LinearAlgebra


function divfunc(u, v, dx)
    u_diff_x = (circshift(u, (-1, 0)) - circshift(u, (0, 0))) / dx;
    v_diff_y = (circshift(v, (0, -1)) - circshift(v, (0, 0))) / dx;
    u_diff_x + v_diff_y
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

    dx = 1/(size(u, 1)-1) |> dev;

    batch_size = size(u, 4);

    u_x = u[:, :, 1, :];
    u_y = u[:, :, 2, :];

    div = divfunc(u_x, u_y, dx);

    div_RHS = reshape(div, size(div)[1]*size(div)[2], batch_size) |> cpu_device();

    possion_matrix_1D = Tridiagonal(
        ones(size(div, 1)-1),
        -4*ones(size(div, 1)),
        ones(size(div, 1)-1)
    );
    possion_matrix_1D = Matrix(possion_matrix_1D);
    possion_matrix_1D[1, end] = 1;
    possion_matrix_1D[end, 1] = 1;

    S = Tridiagonal(
        ones(size(div, 1)-1),
        zeros(size(div, 1)),
        ones(size(div, 1)-1)
    );
    S = Matrix(S);
    S[1, end] = 1;
    S[end, 1] = 1;

    possion_matrix_2D = kron(I(size(div, 2)), possion_matrix_1D) + kron(S, I(size(div, 2)));
    possion_matrix_2D = possion_matrix_2D / (dx * dx);
    possion_matrix_2D = possion_matrix_2D;

    q = zeros(size(possion_matrix_2D, 1), batch_size);
    for i = 1:batch_size
        q[:, i] = possion_matrix_2D \ div_RHS[:, i];
    end
    q = reshape(q, size(div));

    q_grad_x, q_grad_y = gradfunc(q, dx) .|> dev;

    u_x_proj = u_x - q_grad_x;
    u_y_proj = u_y - q_grad_y;

    u_x_proj = reshape(u_x_proj, size(u_x)..., 1);
    u_y_proj = reshape(u_y_proj, size(u_y)..., 1);

    u_x_proj = permutedims(u_x_proj, [1, 2, 4, 3]);
    u_y_proj = permutedims(u_y_proj, [1, 2, 4, 3]);

    return cat(u_x_proj, u_y_proj, dims=3)
end

