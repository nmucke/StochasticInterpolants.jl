
function get_history_state_embedding(;
    in_channels::Int,
    init_channels::Int,
    len_history::Int,
    multiplier::Int=2,
    padding::String="periodic"
)

    if len_history > 0
        return @compact(
                conv = conv_next_block_no_pars(
                    in_channels=len_history*in_channels + in_channels, 
                    out_channels=init_channels, 
                    multiplier=multiplier,
                    padding=padding
                )   
            ) do x
                x, x_0 = x

                H, W, C, len_history, B = size(x_0)
                x_0=reshape(x_0, H, W, C*len_history, B)

                x = cat(x, x_0; dims=3)

                @return conv(x)
            end
    else
        return @compact(
            conv = conv_next_block_no_pars(
                in_channels=len_history*in_channels + in_channels, 
                out_channels=init_channels, 
                multiplier=multiplier,
                padding=padding
            )   
        ) do x
            x, _ = x

            @return conv(x)
        end
    end
end

function get_main_model_with_pars(;
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
            in_channels=model_params["in_channels"], 
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

function get_main_model_no_pars(;
    image_size::Tuple{Int, Int},
    model_params::Dict{Any, Any}
)

    if model_params["model_type"] == "conv_next_u_net"
        return ConvNextUNetNoPars(;
            image_size=image_size,
            out_channels=model_params["out_channels"],
            channels=model_params["channels"], 
            attention_type=model_params["attention_type"],
            use_attention_in_layer=model_params["use_attention_in_layer"],
            attention_embedding_dims=model_params["attention_embedding_dims"],
            num_heads=model_params["num_heads"],
            padding=model_params["padding"],
        )
    end
end

function get_SI_neural_network(;
    image_size::Tuple{Int, Int},
    model_params::Dict{Any, Any}
)

    if model_params["pars_dim"] > 0
        return @compact(
            hist_state_embedding = get_history_state_embedding(;
                in_channels=model_params["in_channels"], 
                init_channels=model_params["channels"][1],
                len_history=model_params["len_history"], 
                multiplier=2, 
                padding=model_params["padding"]
            ),
            t_pars_embedding = get_t_pars_embedding(
                model_params["pars_dim"], true, model_params["embedding_dims"],
            ),
            main_model = get_main_model_with_pars(;
                image_size=image_size,
                model_params=model_params
            )
        ) do x
            x, x_0, pars, t = x

            x = hist_state_embedding((x, x_0))
            t_pars = t_pars_embedding((pars, t))

            @return main_model((x, t_pars))
        end
    else
        return @compact(
            hist_state_embedding = get_history_state_embedding(;
                in_channels=model_params["in_channels"], 
                init_channels=model_params["channels"][1],
                len_history=model_params["len_history"], 
                multiplier=2, 
                padding=model_params["padding"]
            ),
            t_pars_embedding = get_t_pars_embedding(
                model_params["pars_dim"], true, model_params["embedding_dims"],
            ),
            main_model = get_main_model_with_pars(;
                image_size=image_size,
                model_params=model_params
            )
        ) do x
            x, x_0, _, t = x

            x = hist_state_embedding((x, x_0))
            t = t_pars_embedding(t)

            @return main_model((x, t))
        end
    end
end

function get_encoder_neural_network(;
    image_size::Tuple{Int, Int},
    model_params::Dict{Any, Any}
)
    if model_params["pars_dim"] > 0
        return @compact(
            hist_state_embedding = get_history_state_embedding(;
                in_channels=model_params["in_channels"], 
                init_channels=model_params["channels"][1],
                len_history=model_params["len_history"], 
                multiplier=2, 
                padding=model_params["padding"]
            ),
            pars_embedding = get_t_pars_embedding(
                pars_dim, false, model_params["embedding_dims"]
            ),
            main_model = get_main_model_with_pars(;
                image_size=image_size,
                model_params=model_params
            )
        ) do x
            x, x_0, _, pars = x

            x = hist_state_embedding((x, x_0))
            pars = pars_embedding(pars)

            @return main_model((x, pars))
        end
    else
        return @compact(
            hist_state_embedding = get_history_state_embedding(;
                in_channels=model_params["in_channels"], 
                init_channels=model_params["channels"][1],
                len_history=model_params["len_history"], 
                multiplier=2, 
                padding=model_params["padding"]
            ),
            main_model = get_main_model_no_pars(;
                image_size=image_size,
                model_params=model_params
            )
        ) do x
            x, x_0, _, _ = x

            x = hist_state_embedding((x, x_0))

            @return main_model(x)
        end
    end
end