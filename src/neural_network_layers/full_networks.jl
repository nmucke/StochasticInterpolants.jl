
function get_t_pars_embedding(
    pars_dim,
    with_time,
    embedding_dim,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
)

    if pars_dim > 0 && with_time
        pars_embedding_dim = div(embedding_dim, 2)
        t_embedding_dim = div(embedding_dim, 2)
        
        pars_embedding = Chain(
            x -> sinusoidal_embedding(x, min_freq, max_freq, pars_embedding_dim),
            Lux.Dense(pars_embedding_dim => pars_embedding_dim),
            NNlib.gelu,
            Lux.Dense(pars_embedding_dim => pars_embedding_dim),
            NNlib.gelu,
        )

        t_embedding = Chain(
            x -> sinusoidal_embedding(x, min_freq, max_freq, t_embedding_dim),
            Lux.Dense(t_embedding_dim => t_embedding_dim),
            NNlib.gelu,
            Lux.Dense(t_embedding_dim => t_embedding_dim),
            NNlib.gelu,
        )

        t_pars_embedding = Chain(
            (t, pars) -> begin
                t_emb = t_embedding(t)
                pars_emb = pars_embedding(pars)
                return pars_cat(t_emb, pars_emb; dims=1)
            end
        )

    else
        pars_embedding_dim = embedding_dim
        t_pars_embedding = Chain(
            x -> sinusoidal_embedding(x, min_freq, max_freq, pars_embedding_dim),
            Lux.Dense(pars_embedding_dim => pars_embedding_dim),
            NNlib.gelu,
            Lux.Dense(pars_embedding_dim => pars_embedding_dim),
            NNlib.gelu,
        )
    end

    return t_pars_embedding
end

function get_history_embedding(

)


    return 0
end

function get_main_model_with_pars(
    model_type,
    model_params
)

    if model_type == "conv_next_u_net"
        model = 0
    elseif model_type == "diffusion_transformer"
        model = 0
    end

    return 0
end

function get_main_model_no_pars(
    model_type,
    model_params
)

    if model_type == "conv_next_u_net"
        model = 0
    end

    return 0
end



function get_SI_neural_network(
    pars_dim,
    len_history,
    model_type,
    model_params
)

    t_pars_embedding = get_t_pars_embedding(
        pars_dim, true, model_params["embedding_dim"],
    )








    return 0
end


function get_encoder_neural_network(
    pars_dim,
    len_history,
    model_type,
    model_params
)

    if pars_dim > 0
        pars_embedding = get_t_pars_embedding(
            pars_dim, false, model_params["embedding_dim"],
        )
    end



    return 0
end
    