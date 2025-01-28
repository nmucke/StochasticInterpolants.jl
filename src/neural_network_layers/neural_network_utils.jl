using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding
using LinearAlgebra

using Lux.Experimental: @compact 
using Lux.Experimental: @kwdef
using Lux.Experimental: @concrete

export pars_cat, transform_to_nothing, Identity, StateParsIdentity, get_padding, get_attention_layer

function pars_cat(x::AbstractArray, y::Nothing; dims=1)
    return x
end

function pars_cat(x::AbstractArray, y::AbstractArray; dims=1)
    return cat(x, y; dims=dims)
end


function transform_to_nothing(x::AbstractArray)
    return nothing
end

function Identity()
    @compact(
        identity = nothing
    ) do x
        x = x
        @return x
    end
end



function StateParsIdentity()
    @compact(
        identity = nothing
    ) do x
        x, pars = x
        @return x
    end
end

function get_padding(padding::String, padding_size::Int)

    if padding == "constant"
        return a -> pad(a, 0, (padding_size, padding_size))
    elseif padding == "smooth"
        return a -> pad(a, :smooth, (padding_size, padding_size))
    elseif padding == "periodic"
        return a -> pad(a, :periodic, (padding_size, padding_size))
    else
        return a -> a
    end
end



function get_attention_layer(
    attention_type,
    imsize,
    in_channels,
    embed_dim,
    num_heads,
    t_pars_embedding_dims=nothing,
    patch_size=(8, 8)
)

    if attention_type == "linear"
        attention_layer = @compact(
            layer = LinearSpatialAttention(
                imsize=imsize,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        ) do x
            x, t = x
            @return layer(x)
        end
    elseif attention_type == "standard"
        attention_layer = @compact(
            layer = SpatialAttention(
                imsize=imsize,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        ) do x
            x, t = x
            @return layer(x)
        end
    elseif attention_type == "DiT"
        attention_layer = parameter_diffusion_transformer_block(
            in_channels=in_channels,
            out_channels=in_channels,
            pars_dim=t_pars_embedding_dims,
            embed_dim=embed_dim,
            number_heads=num_heads,
            mlp_ratio=2,
            imsize=imsize,
            patch_size=patch_size,
            number_patches=prod(div.(imsize, patch_size))
        )
    else
        attention_layer = Identity()
    end
    
    return attention_layer
end

