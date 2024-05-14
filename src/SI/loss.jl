using Lux
using Random
using CUDA
using NNlib
using Setfield
using StochasticInterpolants
using LuxCUDA


function get_loss(
    x_0::AbstractArray, 
    t::AbstractArray, 
    model::DenoisingDiffusionProbabilisticModel, 
    ps::NamedTuple, 
    st::NamedTuple, 
    rng::AbstractRNG,
    dev=gpu
)

    return 0.0, st
end
