"""
    StochasticInterpolants

Implementation of the Stochastic Interpolants method for generative modeling.
"""
module StochasticInterpolants

using Lux
using Random
using CUDA
using NNlib
using Setfield

include("neural_network_layers.jl")

export residual_block, UNet, UpBlock, DownBlock, sinusoidal_embedding

end
