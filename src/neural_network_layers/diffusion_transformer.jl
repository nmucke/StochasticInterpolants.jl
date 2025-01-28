using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding
using LinearAlgebra
using GPUArraysCore
using StochasticInterpolants

export modulate, ConditionalDiffusionTransformer, parameter_diffusion_transformer_block
export DiffusionTransformerBlock, FinalLayer, DiffusionTransformer

"""
    modulate(x, scale, shift)

Modulate the input tensor `x` by scaling and shifting it.
"""

function modulate(x, scale, shift)

    # Scale
    x = x .* (1.0f0 .+ scale)

    # Shift
    x = x .+ shift    

    return x
end

"""
ConditionalDiffusionTransformer(
    imsize::Dims{2}=(256, 256),
    in_channels::Int=3,
    patch_size::Dims{2}=(16, 16),
    embed_planes::Int=768,
    depth::Int=6,
    number_heads=16,
    mlp_ratio=4.0f0,
    dropout_rate=0.1f0,
    embedding_dropout_rate=0.1f0,
    pool::Symbol=:class,
    num_classes::Int=1000,
    kwargs...
)

Create a DiffusionTransformer Transformer model with the given image size, number of input
channels, patch size, embedding planes, depth, number of heads, MLP ratio,
dropout rate, embedding dropout rate, pooling method, number of classes, and
additional keyword arguments.

Based on https://github.com/LuxDL/Boltz.jl/blob/v0.3.9/src/vision/vit.jl#L48-L61
The architecture is based on https://arxiv.org/abs/2212.09748
"""
struct ConditionalDiffusionTransformer <: Lux.AbstractExplicitContainerLayer{
    (:t_embedding, :pars_embedding, :patchify_layer, :positional_embedding, :dit_blocks, :final_layer)
}
    t_embedding::Lux.AbstractExplicitLayer
    pars_embedding::Lux.AbstractExplicitLayer
    patchify_layer::Lux.AbstractExplicitLayer
    positional_embedding::Lux.AbstractExplicitLayer
    dit_blocks::Lux.AbstractExplicitLayer
    final_layer::Lux.AbstractExplicitLayer
    unpatchify::Function
    len_history::Int
end

function ConditionalDiffusionTransformer(
    imsize::Tuple{Int, Int};
    in_channels::Int=3, 
    out_channels=nothing,
    patch_size::Tuple{Int, Int}=(16, 16),
    embedding_dims::Int=768, 
    depth::Int=6, 
    number_heads=16,
    mlp_ratio=4.0f0, 
    # dropout_rate=0.1f0, 
    # embedding_dropout_rate=0.1f0,
    pars_dim=128,
    len_history=1,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
)

    if isnothing(out_channels)
        out_channels = in_channels
    end

    number_patches = prod(imsize .÷ patch_size)

    if pars_dim == 0
        t_pars_embedding_dims = embedding_dims

        pars_embedding = Chain(
            a -> transform_to_nothing(a),
        )
    else
        t_pars_embedding_dims = div(embedding_dims, 2)

        pars_embedding = Chain(
            x -> sinusoidal_embedding(x, min_freq, max_freq, t_pars_embedding_dims),
            Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
            NNlib.gelu,
            Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
        )
    end

    t_embedding = Chain(
        x -> sinusoidal_embedding(x, min_freq, max_freq, t_pars_embedding_dims),
        Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
        NNlib.gelu,
        Lux.Dense(t_pars_embedding_dims => t_pars_embedding_dims),
    )

    init_channels = in_channels * len_history + in_channels

    patchify_layer = patchify(
        imsize; 
        in_channels=init_channels, 
        patch_size=patch_size, 
        embed_planes=embedding_dims
    )

    positional_embedding = ViPosEmbedding(embedding_dims, number_patches)

    dit_blocks = Chain(
        [DiffusionTransformerBlock(embedding_dims, number_heads, mlp_ratio) for _ in 1:depth]...;
        disable_optimizations=true
    )

    final_layer = FinalLayer(embedding_dims, patch_size, out_channels) # (E, patch_size ** 2 * out_channels, N)

    h = div(imsize[1], patch_size[1])
    w = div(imsize[2], patch_size[2])
    _unpatchify(x) = unpatchify(x, (h, w), patch_size, out_channels)

    return ConditionalDiffusionTransformer(
        t_embedding, pars_embedding, patchify_layer, 
        positional_embedding, dit_blocks, final_layer, 
        _unpatchify, len_history
    )
end

function (dt::ConditionalDiffusionTransformer)(
    x, 
    ps::NamedTuple,
    st::NamedTuple
)

    x, x_0, pars, t = x

    t, st_new = dt.t_embedding(t, ps.t_embedding, st.t_embedding)
    @set! st.t_embedding = st_new

    pars, st_new = dt.pars_embedding(pars, ps.pars_embedding, st.pars_embedding)
    @set! st.pars_embedding = st_new

    t_pars_embed = pars_cat(t, pars; dims=1)

    H, W, C, len_history, B = size(x_0)
    x_0 = reshape(x_0, H, W, C*len_history, B);
    x = cat(x, x_0; dims=3)

    x, st_new = dt.patchify_layer(x, ps.patchify_layer, st.patchify_layer)
    @set! st.patchify_layer = st_new

    x, st_new = dt.positional_embedding(x, ps.positional_embedding, st.positional_embedding)
    @set! st.positional_embedding = st_new

    for i in 1:length(dt.dit_blocks)
        dit_layer_name = Symbol(:layer_, i)
        x, new_st = dt.dit_blocks[i](
            (x, t_pars_embed), ps.dit_blocks[dit_layer_name], st.dit_blocks[dit_layer_name]
        )  
        @set! st.dit_blocks[dit_layer_name] = new_st
    end

    x, st_new = dt.final_layer((x, t_pars_embed), ps.final_layer, st.final_layer)
    @set! st.final_layer = st_new

    x = dt.unpatchify(x)

    return x, st
end

###############################################################################
# Parameter Diffusion Transformer Block
###############################################################################

"""
    parameter_diffusion_transformer_block(
        in_channels::Int,
        out_channels::Int,
        pars_dim::Int,
        embed_dim::Int,
        number_heads::Int,
        mlp_ratio::Int,
        imsize::Tuple{Int, Int},
        patch_size::Int,
        number_patches::Int
    )

Create a parameter diffusion transformer block with the given number of input
and output channels, parameter dimensions, embedding dimensions, number of heads,
MLP ratio, image size, patch size, and number of patches.
"""
function parameter_diffusion_transformer_block(;
    in_channels::Int,
    out_channels::Int,
    pars_dim::Int,
    embed_dim::Int,
    number_heads::Int,
    mlp_ratio::Int,
    imsize::Tuple{Int, Int},
    patch_size::Tuple{Int, Int},
    number_patches::Int
)
    @compact(

        pars_embedding = Lux.Chain(
            Lux.Dense(pars_dim => embed_dim),
        ),
        patchify_layer = patchify(
            imsize; 
            in_channels=in_channels, 
            patch_size=patch_size, 
            embed_planes=embed_dim
        ),
        positional_embedding = ViPosEmbedding(embed_dim, number_patches),
        dit_block = DiffusionTransformerBlock(embed_dim, number_heads, mlp_ratio),
        final_layer = Chain(
            Lux.LayerNorm((embed_dim, 1); affine=false),
            Lux.Dense(embed_dim => patch_size[1] * patch_size[2] * out_channels)
        ),
    ) do x
        x, pars = x

        pars = pars_embedding(pars)

        x = patchify_layer(x)

        x = positional_embedding(x)

        x = dit_block((x, pars))

        x = final_layer(x)

        h = div(imsize[1], patch_size[1])
        w = div(imsize[2], patch_size[2])

        @return unpatchify(x, (h, w), patch_size, out_channels)
    end
end


"""
    DiffusionTransformerBlock(
        in_channels::Int, 
        out_channels::Int,
        block_depth::Int
    )
    
Create a diffusion transformer block
"""
struct DiffusionTransformerBlock <: Lux.AbstractExplicitContainerLayer{
    (:layer_norm_1, :layer_norm_2, :attention, :mlp, :adaLN_modulation)
}
    layer_norm_1::Lux.LayerNorm
    layer_norm_2::Lux.LayerNorm
    attention::Lux.AbstractExplicitLayer
    mlp::Lux.Chain
    adaLN_modulation::Lux.Dense
    hidden_size::Int
    #reshape_modulation::Function
end

function DiffusionTransformerBlock(
    hidden_size::Int, 
    num_heads::Int,
    mlp_ratio=4.0f0,    
)

    layer_norm_1 = Lux.LayerNorm((hidden_size, 1); affine=true)
    layer_norm_2 = Lux.LayerNorm((hidden_size, 1); affine=true)

    attention = MultiHeadSelfAttention(
        hidden_size, num_heads; 
        attention_dropout_rate=0.1f0,
        projection_dropout_rate=0.1f0
    )

    mlp_hidden_dim = floor(Int, mlp_ratio * hidden_size)
    mlp = Lux.Chain(
        Lux.Dense(hidden_size => mlp_hidden_dim, NNlib.gelu),
        Lux.Dropout(0.1f0),
        Lux.Dense(mlp_hidden_dim => hidden_size),
    )

    adaLN_modulation = Lux.Dense(hidden_size => 6 * hidden_size)

    return DiffusionTransformerBlock(
        layer_norm_1, 
        layer_norm_2, 
        attention, 
        mlp, 
        adaLN_modulation,
        hidden_size,
    )
end

function (dit_block::DiffusionTransformerBlock)(
    x, 
    ps,
    st::NamedTuple
)


    x, c = x

    modulation, st_new = dit_block.adaLN_modulation(c, ps.adaLN_modulation, st.adaLN_modulation)
    @set! st.adaLN_modulation = st_new

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _fast_chunk(modulation, Val(6), Val(1))

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = reshape_modulation.(
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), size(x, 2)
    )

    x, st_new = dit_block.layer_norm_1(x, ps.layer_norm_1, st.layer_norm_1)
    @set! st.layer_norm_1 = st_new
    x = modulate(x, scale_msa, shift_msa)

    attn, st_new = dit_block.attention(x, ps.attention, st.attention)
    @set! st.attention = st_new
    x = x .+ gate_msa .* attn

    x, st_new = dit_block.layer_norm_2(x, ps.layer_norm_2, st.layer_norm_2)
    @set! st.layer_norm_2 = st_new

    x = modulate(x, scale_mlp, shift_mlp)

    mlp_out, st_new = dit_block.mlp(x, ps.mlp, st.mlp)
    @set! st.mlp = st_new

    x = x .+ gate_mlp .* mlp_out

    return x, st
end


"""
    FinalLayer(
        hidden_size::Int, 
        patch_size::Int, 
        out_channels::Int
    )

Create the final layer of the diffusion transformer
"""

struct FinalLayer <: Lux.AbstractExplicitContainerLayer{
    (:norm_final, :dense, :adaLN_modulation)
}
    norm_final::Lux.LayerNorm
    dense::Lux.Dense
    adaLN_modulation::Lux.Chain
end

function FinalLayer(
    hidden_size::Int, 
    patch_size::Tuple{Int, Int}, 
    out_channels::Int
)

    norm_final = Lux.LayerNorm((hidden_size, 1); affine=false)
    adaLN_modulation = Lux.Chain(
        NNlib.gelu,
        Lux.Dense(hidden_size => 2 * hidden_size)
    )
    dense = Lux.Dense(hidden_size => patch_size[1] * patch_size[2] * out_channels)

    return FinalLayer(norm_final, dense, adaLN_modulation)
end

function (final_layer::FinalLayer)(
    input,
    ps,
    st::NamedTuple
)

    x, t = input

    modulation, st_new = final_layer.adaLN_modulation(t, ps.adaLN_modulation, st.adaLN_modulation)
    @set! st.adaLN_modulation = st_new

    shift, scale = _fast_chunk(modulation, Val(2), Val(1))
    shift, scale = reshape_modulation.((shift, scale), size(x, 2))

    x, st_new = final_layer.norm_final(x, ps.norm_final, st.norm_final)
    @set! st.norm_final = st_new

    x = modulate(x, scale, shift)

    x, st_new = final_layer.dense(x, ps.dense, st.dense)
    @set! st.dense = st_new

    return x, st
end

    
    
"""
    DiffusionTransformer(
        imsize::Dims{2}=(256, 256),
        in_channels::Int=3,
        patch_size::Dims{2}=(16, 16),
        embed_planes::Int=768,
        depth::Int=6,
        number_heads=16,
        mlp_ratio=4.0f0,
        dropout_rate=0.1f0,
        embedding_dropout_rate=0.1f0,
        pool::Symbol=:class,
        num_classes::Int=1000,
        kwargs...
    )

Create a DiffusionTransformer Transformer model with the given image size, number of input
channels, patch size, embedding planes, depth, number of heads, MLP ratio,
dropout rate, embedding dropout rate, pooling method, number of classes, and
additional keyword arguments.

Based on https://github.com/LuxDL/Boltz.jl/blob/v0.3.9/src/vision/vit.jl#L48-L61
The architecture is based on https://arxiv.org/abs/2212.09748
"""
struct DiffusionTransformer <: Lux.AbstractExplicitContainerLayer{
    (:patchify_layer, :positional_embedding, :dit_blocks, :final_layer)#, :conv)
}
    patchify_layer::Lux.Chain
    positional_embedding::Lux.AbstractExplicitLayer
    dit_blocks::Lux.AbstractExplicitLayer
    final_layer::Lux.AbstractExplicitLayer
    unpatchify::Function
    
end

function DiffusionTransformer(;
    image_size::Tuple{Int, Int},
    in_channels::Int=3, 
    out_channels=nothing,
    patch_size::Tuple{Int, Int}=(16, 16),
    embedding_dims::Int=768, 
    depth::Int=6, 
    num_heads=16,
    mlp_ratio=4.0f0, 
    dropout_rate=0.1f0, 
    embedding_dropout_rate=0.1f0,
)

    if isnothing(out_channels)
        out_channels = in_channels
    end

    number_patches = prod(image_size .÷ patch_size)

    patchify_layer = patchify(
        image_size; 
        in_channels=in_channels, 
        patch_size=patch_size, 
        embed_planes=embedding_dims
    )

    positional_embedding = ViPosEmbedding(embedding_dims, number_patches)

    dit_blocks = Chain(
        [DiffusionTransformerBlock(embedding_dims, num_heads, mlp_ratio) for _ in 1:depth]...;
    )

    final_layer = Chain(
        Lux.LayerNorm((embedding_dims, 1); affine=false),
        Lux.Dense(embedding_dims => patch_size[1] * patch_size[2] * out_channels)
    ) # (patch_size ** 2 * out_channels, N)

    # FinalLayer(embedding_dims, patch_size, out_channels) # (E, patch_size ** 2 * out_channels, N)

    h = div(image_size[1], patch_size[1])
    w = div(image_size[2], patch_size[2])
    _unpatchify(x) = unpatchify(x, (h, w), patch_size, out_channels)

    return DiffusionTransformer(patchify_layer, positional_embedding, dit_blocks, final_layer, _unpatchify)
end

function (dt::DiffusionTransformer)(
    input, 
    ps::NamedTuple,
    st::NamedTuple
)

    x, t = input

    x, st_new = dt.patchify_layer(x, ps.patchify_layer, st.patchify_layer)
    @set! st.patchify_layer = st_new

    x, st_new = dt.positional_embedding(x, ps.positional_embedding, st.positional_embedding)
    @set! st.positional_embedding = st_new


    for i in 1:length(dt.dit_blocks)
        layer_name = Symbol(:layer_, i)

        x, new_st = dt.dit_blocks[i](
            (x, t), ps.dit_blocks[layer_name], st.dit_blocks[layer_name]
        )
        @set! st.dit_blocks[layer_name] = new_st
    end

    x, st_new = dt.final_layer(x, ps.final_layer, st.final_layer)
    @set! st.final_layer = st_new

    x = dt.unpatchify(x)

    return x, st
end





