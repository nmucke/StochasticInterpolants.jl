

using IncompressibleNavierStokes
using DSP

export dissipative_operator
export quadratic_operator
export PhysicsConsistentModel

function transpose_kernel(kernel)
    # Get B kernel and its transpose
    kernel = reverse(kernel, dims=1:2)
    return kernel[:, :, [1, 3, 2, 4], :]
end

function velocity_kernel_product(x, B_kernel)
    return cat(
        NNlib.conv(x[:, :, 1:1, :], B_kernel[:, :, 1, :, :]) + NNlib.conv(x[:, :, 2:2, :], B_kernel[:, :, 2, :, :]), 
        NNlib.conv(x[:, :, 1:1, :], B_kernel[:, :, 3, :, :]) + NNlib.conv(x[:, :, 2:2, :], B_kernel[:, :, 4, :, :]); 
        dims=3
    )
end


"""
    dissipative_operator(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        padding="constant"
    )

    out = Q(weights)^T * Q = B(weights)^T * q(u)^2 * B(weights)

    where Q = q(u) * B and B is a convolutional block.
"""
function dissipative_operator(;
    in_channels::Int, 
    out_channels::Int,
    pars_embed_dim::Int,
    multiplier::Int = 1,
    kernel_size::Int = 3,
    padding="constant",
)

    @compact(
        padding = get_padding(padding, div(kernel_size, 2)),
        q_block = @compact(
            block_1 = conv_next_block_no_pars(
                in_channels=in_channels, 
                out_channels=4*in_channels, 
                multiplier=multiplier,
                padding=padding
            ),
            block_2 = conv_next_block_no_pars(
                in_channels=4*in_channels, 
                out_channels=out_channels, 
                multiplier=multiplier,
                padding=padding
            )
        ) do x
            x = block_1(x)
            @return block_2(x)
        end,
        B_kernel = kaiming_normal(kernel_size, kernel_size, out_channels^2, 1, 1),
    ) do x
        x = x
        
        # Compute q squared
        q = q_block(x) # [H, W, C=2, B]
        q_squared = q.^2

        # Get B kernel transpose
        B_kernel_transpose = transpose_kernel(B_kernel)

        x = padding(x)

        # Compute B * x
        Bx = velocity_kernel_product(x, B_kernel)

        # Compute q^2 .* B * x
        q_squared_Bx = q_squared .* Bx

        q_squared_Bx = padding(q_squared_Bx)

        # Compute B^T * q^2 .* B * x
        out = velocity_kernel_product(q_squared_Bx, B_kernel_transpose)

        @return out
    end
end


"""
    skew_symmetric_operator(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        padding="constant"
    )

    out = 
"""
function skew_symmetric_operator(;
    in_channels::Int, 
    out_channels::Int,
    kernel_size::Int = 3,
    padding="constant"
)
    return @compact(
        padding = get_padding(padding, div(kernel_size, 2)),
        q_block = Lux.Conv((kernel_size, kernel_size), (in_channels => in_channels); use_bias=false),
        B_1_kernel = kaiming_normal(kernel_size, kernel_size, out_channels^2, 1, 1),
        B_2_kernel = kaiming_normal(kernel_size, kernel_size, out_channels^2, 1, 1),
    ) do x

        x = padding(x)

        # Get B kernel transpose
        B_1_kernel_transpose = transpose_kernel(B_1_kernel)
        B_2_kernel_transpose = transpose_kernel(B_2_kernel)

        # Compute q squared
        # q = q_block((x, pars)) # [H, W, C=2, B]
        q = q_block(x) # [H, W, C=2, B]

        # Compute K * x
        Kx = velocity_kernel_product(x, B_2_kernel)
        Kx = q .* Kx
        Kx = padding(Kx)
        Kx = velocity_kernel_product(Kx, B_1_kernel_transpose)
        
        # Compute K^T * x
        KTx = velocity_kernel_product(x, B_1_kernel)
        KTx = q .* KTx
        KTx = padding(KTx)
        KTx = velocity_kernel_product(KTx, B_2_kernel_transpose)

        @return Kx - KTx
    end
end

"""
    quadratic_operator(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        padding="constant"
    )

    The operator is such that Q(weights)^T * Q = B(weights)^T * q(u)^2 * B(weights)

    where Q = q(u) * B and B is a convolutional block.
"""
function quadratic_operator(;
    Re = 1000f0,
    image_size::Tuple{Int, Int}=(128, 128),
)

    # Setup
    ax_x = LinRange(0f0, 2f0*pi, image_size[1] + 1) 
    ax_y = LinRange(0f0, 2f0*pi, image_size[2] + 1)
    setup = Setup(; x = (ax_x, ax_y), Re, ArrayType = CuArray{Float32});

    create_right_hand_side(setup, psolver) = function right_hand_side(u)
        (; Iu) = setup.grid
        u = pad_circular(u, 1; dims = 1:2)
        out = Array{eltype(u)}[]
        for i in 1:size(u, 4)
            t = zero(eltype(u))
            u_ = u[:, :, 1, i], u[:, :, 2, i]
            F = IncompressibleNavierStokes.momentum(u_, nothing, t, setup)
            # F = IncompressibleNavierStokes.convection(u_, setup)
            F = cat(F[1], F[2]; dims = 3)
            F = F[2:end-1, 2:end-1, :]
            out = [out; [F]]
        end
        stack(out; dims = 4)
    end

    f = create_right_hand_side(setup, nothing);
    
    physics_term(x) = begin
        return f(x)
    end
    
    return physics_term
end


function _get_forcing_model(
    image_size::Tuple{Int, Int},
    model_params::Dict{Any, Any}
)
    if model_params["model_type"] == "conv_next_u_net"
        return ConvNextUNetWithPars(;
            image_size=image_size,
            out_channels=model_params["out_channels"],
            channels=model_params["channels"], 
            attention_type=model_params["attention_type"],
            use_attention_in_layer=model_params["use_attention_in_layer"],
            attention_embedding_dims=model_params["attention_embedding_dims"],
            embedding_dims=model_params["embedding_dims"],
            num_heads=model_params["num_heads"],
            padding=model_params["padding"],
        )
    elseif model_params["model_type"] == "diffusion_transformer"
        return DiffusionTransformer(;
            image_size=image_size,
            in_channels=model_params["in_channels"] * model_params["len_history"] + model_params["in_channels"], 
            out_channels=model_params["out_channels"],
            patch_size=(model_params["patch_size"], model_params["patch_size"]),
            embedding_dims=model_params["embedding_dims"], 
            depth=model_params["depth"], 
            num_heads=model_params["num_heads"],
            mlp_ratio=model_params["mlp_ratio"], 
            dropout_rate=model_params["dropout_rate"], 
            embedding_dropout_rate=model_params["embedding_dropout_rate"],
        )
    end
end

"""
    physics_consistent_network(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        padding="constant"
    )

    out = 
"""
function PhysicsConsistentModel(
    image_size::Tuple{Int, Int},
    model_params
)
    return @compact(
        t_pars_embedding = get_t_pars_embedding(
            model_params["pars_dim"], true, model_params["embedding_dims"],
        ),
        hist_state_embedding = get_history_state_embedding(;
            in_channels=model_params["forcing"]["in_channels"], 
            init_channels=model_params["forcing"]["channels"][1],
            len_history=model_params["len_history"], 
            multiplier=2, 
            padding=model_params["forcing"]["padding"]
        ),
        dissipation = dissipative_operator(
            in_channels=model_params["dissipation"]["in_channels"], 
            out_channels=model_params["dissipation"]["out_channels"],
            pars_embed_dim=model_params["dissipation"]["pars_embed_dim"],
            multiplier=model_params["dissipation"]["multiplier"],
            kernel_size=model_params["dissipation"]["kernel_size"],
            padding=model_params["dissipation"]["padding"]
        ),
        quadratic = skew_symmetric_operator(
            in_channels=2, 
            out_channels=2,
            kernel_size=5,
            padding="periodic"
        ),
        # quadratic_operator(;
        #     Re = 1000f0,
        #     image_size=image_size,
        # ),
        forcing = _get_forcing_model(image_size, model_params["forcing"])
    
    ) do x
        x, x_0, pars, t = x

        # Compute t pars embedding
        t_pars = t_pars_embedding(t)

        # Compute dissipation
        dissipation_term = dissipation(x)

        # Compute quadratic
        quadratic_term = quadratic(x)

        # Compute forcing
        x = hist_state_embedding((x, x_0))

        forcing_term = forcing((x, t_pars))

        @return forcing_term + quadratic_term - dissipation_term
    end





end