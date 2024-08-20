using Lux, Random
using Lux.Experimental: @compact
using StochasticInterpolants

function dit_down_block(
    imsize::Tuple{Int, Int};
    in_channels::Int=3, 
    out_channels=nothing,
    patch_size::Tuple{Int, Int}=(16, 16),
    embed_dim::Int=768, 
    depth::Int=1, 
    number_heads=4,
    mlp_ratio=4.0f0, 
    dropout_rate=0.1f0, 
    embedding_dropout_rate=0.1f0,
    pars_dim=128
)
    @compact(
        down_block = DownBlock(in_channels, out_channels, 1),
        DiT = DiffusionTransformer(
            div.(imsize, 2);
            in_channels=in_channels, 
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim, 
            depth=1, 
            number_heads=number_heads,
            mlp_ratio=mlp_ratio, 
            pars_dim=pars_dim
        )        
    ) do x
        x, pars = x
        (x, skips) = down_block(x)
        @return DiT((x, pars)), skips
    end
end

function dit_up_block(
    imsize::Tuple{Int, Int};
    in_channels::Int=3, 
    out_channels=nothing,
    patch_size::Tuple{Int, Int}=(16, 16),
    embed_dim::Int=768, 
    depth::Int=1, 
    number_heads=4,
    mlp_ratio=4.0f0, 
    dropout_rate=0.1f0, 
    embedding_dropout_rate=0.1f0,
    with_conv=false,
    temporal_embedding=false,
    pars_dim=128
)
    @compact(
        up_block = UpBlock(in_channels, out_channels, 1),
        DiT = DiffusionTransformer(
            imsize .* 2;
            in_channels=in_channels, 
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim, 
            depth=1, 
            number_heads=number_heads,
            mlp_ratio=mlp_ratio, 
            pars_dim=pars_dim
        )        
    ) do x
        x, pars = x
        x = up_block(x)
        @return DiT((x, pars))
    end
end











"""
    ConditionalTransformerUNet(
        image_size::Tuple{Int, Int},
        channels::Vector{Int} = [32, 64, 96, 128],
        block_depth::Int = 2,
        min_freq::Float32 = 1.0f0,
        max_freq::Float32 = 1000.0f0,
        embedding_dims::Int = 32
    )

Create a conditional U-Net model with the given image size, number of channels,
block depth, frequency range, and embedding dimensions.
"""
struct ConditionalTransformerUNet <: Lux.AbstractExplicitContainerLayer{
    (:upsample, :conv_in, :init_conv_in, :conv_out, :down_blocks, :residual_blocks, :up_blocks)
}
    upsample::Upsample
    conv_in::Conv
    init_conv_in::Conv
    conv_out::Conv
    down_blocks::Lux.AbstractExplicitLayer
    residual_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    noise_embedding::Function
end

function ConditionalTransformerUNet(
    image_size::Tuple{Int, Int}; 
    in_channels::Int = 3,
    channels=[32, 64, 96, 128], 
    block_depth=2,
    min_freq=1.0f0, 
    max_freq=1000.0f0, 
    embedding_dims=32
)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1), in_channels => embedding_dims)
    init_conv_in = Conv((1, 1), in_channels => embedding_dims)
    conv_out = Conv((1, 1), channels[1] => in_channels; init_weight=Lux.zeros32)

    t_embedding = x -> sinusoidal_embedding(x, min_freq, max_freq, embedding_dims)

    channel_input = embedding_dims + embedding_dims + embedding_dims #channels[1] + channels[1]

    down_blocks = []
    push!(down_blocks, DownBlock(channel_input, channels[1], block_depth))
    for i in 1:(length(channels) - 2)
        push!(down_blocks, DownBlock(channels[i], channels[i + 1], block_depth))
    end
    down_blocks = Chain(down_blocks...; disable_optimizations=true)

    
    residual_blocks = []
    push!(residual_blocks, residual_block(channels[end - 1], channels[end]))
    for _ in 2:block_depth
        push!(residual_blocks, residual_block(channels[end], channels[end]))
    end
    residual_blocks = Chain(residual_blocks...; disable_optimizations=true)

    reverse!(channels)
    up_blocks = [UpBlock(channels[i], channels[i + 1], block_depth)
                 for i in 1:(length(channels) - 1)]
    up_blocks = Chain(up_blocks...)

    return ConditionalTransformerUNet(upsample, conv_in, init_conv_in, conv_out, down_blocks, residual_blocks, up_blocks, noise_embedding)
end


function (cond_trans_unet::ConditionalTransformerUNet)(
    x,#::Tuple{AbstractArray{T, 4}, AbstractArray{T, 4}, AbstractArray{S, 4}}, 
    ps::NamedTuple,
    st::NamedTuple
)# where T <: AbstractFloat

    x, x0, pars, t = x

    t = reshape(t, (1, 1, 1, 1, size(t, 1)))

    t_embed = t_embedding(t)

    noisy_images, init_noisy_images, noise_variances = x
    @assert size(noise_variances)[1:3] == (1, 1, 1)
    @assert size(noisy_images, 4) == size(noise_variances, 4)
    @assert size(init_noisy_images, 4) == size(noise_variances, 4)

    emb = conditional_unet.noise_embedding(noise_variances)
    @assert size(emb)[[1, 2, 4]] == (1, 1, size(noise_variances, 4))
    emb, _ = conditional_unet.upsample(emb, ps.upsample, st.upsample)
    @assert size(emb)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noise_variances, 4))

    x, new_st = conditional_unet.conv_in(noisy_images, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    init_x, new_st = conditional_unet.init_conv_in(init_noisy_images, ps.init_conv_in, st.init_conv_in)
    @set! st.init_conv_in = new_st
    @assert size(init_x)[[1, 2, 4]] ==
            (size(init_noisy_images, 1), size(init_noisy_images, 2), size(init_noisy_images, 4))

    x = cat(x, init_x, emb; dims=3)
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    skips_at_each_stage = ()
    for i in 1:length(conditional_unet.down_blocks)
        layer_name = Symbol(:layer_, i)
        (x, skips), new_st = conditional_unet.down_blocks[i](x, ps.down_blocks[layer_name],
                                                 st.down_blocks[layer_name])
        #x = leakyrelu.(x)                                              
        @set! st.down_blocks[layer_name] = new_st
        skips_at_each_stage = (skips_at_each_stage..., skips)
    end

    x, new_st = conditional_unet.residual_blocks(x, ps.residual_blocks, st.residual_blocks)
    @set! st.residual_blocks = new_st

    for i in 1:length(conditional_unet.up_blocks)
        layer_name = Symbol(:layer_, i)
        x, new_st = conditional_unet.up_blocks[i]((x, skips_at_each_stage[end - i + 1]),
                                      ps.up_blocks[layer_name], st.up_blocks[layer_name])
        #x = leakyrelu.(x)
        @set! st.up_blocks[layer_name] = new_st
    end

    x, new_st = conditional_unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end






















































































































































# """
#     multihead_self_attention(
#         input_dim, 
#         output_dim, 
#         hidden_dim,
#         num_heads
#     )

# Constructs a multihead self-attention layer. Argument dimension is: `(input_dim, seq_len, batch_size)`.
# """
# function multihead_self_attention(
#     input_dim, 
#     output_dim, 
#     hidden_dim,
#     num_heads
# )
#     @compact(
#         out = Dense(num_heads * div(hidden_dim, num_heads) => output_dim),
#         heads = [
#             @compact(
#                 K = Dense(input_dim => div(hidden_dim, num_heads); use_bias=false),
#                 V = Dense(input_dim => div(hidden_dim, num_heads); use_bias=false),
#                 Q = Dense(input_dim => div(hidden_dim, num_heads); use_bias=false)
#             ) do x
#                 k, v, q = K(x), V(x), Q(x)
#                 x = sum(k .* q; dims=1) ./ div(hidden_dim, num_heads)
#                 softmax(x; dims=2) .* v
#             end for _ in 1:num_heads
#         ]
#     ) do x
#         out(reduce(vcat, [h(x) for h in heads]))
#     end
# end

# """
#     multihead_cross_attention(
#         input_dim, 
#         output_dim, 
#         hidden_dim,
#         num_heads
#     )

# Constructs a multihead cross-attention layer. Argument dimensions are:
# x: `(input_dim, seq_len, batch_size)`
# y: `(input_dim, seq_len, batch_size)`
# """
# function multihead_cross_attention(
#     input_dim, 
#     output_dim, 
#     hidden_dim,
#     num_heads
# )
#     @compact(
#         out = Dense(num_heads * div(hidden_dim, num_heads) => output_dim),
#         heads = [
#             @compact(
#                 K = Dense(input_dim => div(hidden_dim, num_heads); use_bias=false),
#                 V = Dense(input_dim => div(hidden_dim, num_heads); use_bias=false),
#                 Q = Dense(input_dim => div(hidden_dim, num_heads); use_bias=false)
#             ) do x
#                 x, y = x
#                 k, v, q = K(y), V(y), Q(x)
#                 x = sum(k .* q; dims=1) ./ div(hidden_dim, num_heads)
#                 softmax(x; dims=2) .* v
#             end for _ in 1:num_heads
#         ]
#     ) do x
#         out(reduce(vcat, [h(x) for h in heads]))
#     end
# end


# function transformer_block(
#     embedding_dim,
#     num_heads,
#     mlp_dim
# )
#     @compact(
#         attn_self = multihead_self_attention(embedding_dim, embedding_dim, embedding_dim, num_heads),
#         attn_cross = multihead_cross_attention(embedding_dim, embedding_dim, embedding_dim, num_heads),
#         norm1 = Lux.LayerNorm((embedding_dim, 1)),
#         norm2 = Lux.LayerNorm((embedding_dim, 1)),
#         norm3 = Lux.LayerNorm((embedding_dim, 1)),
#         ffn = Chain(
#             Dense(embedding_dim => mlp_dim),
#             NNlib.gelu,
#             Dense(mlp_dim => embedding_dim)
#         )
#     ) do x
#         x, y = x
#         x = attn_self(norm1(x)) .+ x
#         x = attn_cross((norm2(x), y)) .+ x
#         ffn(norm3(x)) .+ x
#     end
# end



# """
#     patchify(
#         imsize::Tuple{Int, Int}=(64, 64),
#         in_channels::Int=3,
#         patch_size::Tuple{Int, Int}=(8, 8),
#         embed_planes::Int=128,
#         norm_layer=Returns(Lux.NoOpLayer()),
#         flatten=true
#     )

# Create a patch embedding layer with the given image size, number of input
# channels, patch size, embedding planes, normalization layer, and flatten flag.

# Based on https://github.com/LuxDL/Boltz.jl/blob/v0.3.9/src/vision/vit.jl#L48-L61
# """
# function patchify(
#     imsize::Tuple{Int, Int}; 
#     in_channels=3, 
#     patch_size=(8, 8),
#     embedding_dim=128, 
#     norm_layer=Returns(Lux.NoOpLayer()), 
#     flatten=true
# )

#     im_width, im_height = imsize
#     patch_width, patch_height = patch_size

#     @assert (im_width % patch_width == 0) && (im_height % patch_height == 0)

#     return Lux.Chain(Lux.Conv(patch_size, in_channels => embedding_dim; stride=patch_size),
#         flatten ? Boltz._flatten_spatial : identity, norm_layer(embedding_dim))
# end

# """
#     unpatchify(x, patch_size, out_channels)

# Unpatchify the input tensor `x` with the given patch size and number of output
# channels.
# """

# function unpatchify(x, patch_size, out_channels)
    
#     c = out_channels
#     p = patch_size[1]
#     h = w = round(Int, sqrt(size(x, 2)))
#     @assert h * w == size(x, 2)

#     x = reshape(x, (p, p, c, h, w, size(x, 3)))
#     x = permutedims(x, (1, 4, 2, 5, 3, 6))
#     imgs = reshape(x, (h * p, h * p, c, size(x, 6)))
#     return imgs
# end

