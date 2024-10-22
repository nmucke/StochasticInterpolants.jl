using Lux
using Random
using CUDA
using NNlib
using Setfield
using ArrayPadding
using LinearAlgebra
using GPUArraysCore
using StochasticInterpolants


 
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
    # conv::Lux.Chain

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

    # x, st_new = dt.dit_blocks((x, t_pars_embed), ps.dit_blocks, st.dit_blocks)
    # @set! st.dit_blocks = st_new

    x, st_new = dt.final_layer((x, t_pars_embed), ps.final_layer, st.final_layer)
    @set! st.final_layer = st_new

    x = dt.unpatchify(x)

    return x, st
end
