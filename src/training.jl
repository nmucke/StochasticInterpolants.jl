using Lux
using StochasticInterpolants
using Random
using CUDA
using NNlib
using Setfield
using MLDatasets
using Plots
using Optimisers
using Zygote
using LuxCUDA
using Images
using ImageFiltering
using Printf
using Statistics



function train_diffusion_model(
    model,
    ps::NamedTuple,
    st::NamedTuple,
    opt_state::NamedTuple,
    trainset::AbstractArray,
    num_epochs::Int,
    batch_size::Int,
    num_samples::Int,
    rng::AbstractRNG,
    dev=gpu
)
    cpu_dev = LuxCPUDevice();

    num_train = size(trainset)[end] 
    for epoch in 1:num_epochs
        running_loss = 0.0

        # Shuffle trainset
        train_ids = shuffle(rng, 1:num_train)
        trainset = trainset[:, :, :, train_ids]
    
        for i in 1:batch_size:size(trainset)[end]
            
            if i + batch_size - 1 > size(trainset)[end]
                break
            end

            x = trainset[:, :, :, i:i+batch_size-1] |> dev;

            if typeof(model) == ScoreMatchingLangevinDynamics
                t = rand(rng, Float32, (batch_size,)) |> dev;

            elseif typeof(model) == DenoisingDiffusionProbabilisticModel
                t = rand(rng, 0:model.timesteps-1, (batch_size,)) |> dev;
            end

            (loss, st), pb_f = Zygote.pullback(
                p -> get_loss(x, t, model, p, st, rng, dev), ps
            );
            running_loss += loss

            gs = pb_f((one(loss), nothing))[1];
            
            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
            
            # GC.gc()

            if i % 1 == 0
                CUDA.reclaim()
            end
            

        end
        
        running_loss /= floor(Int, size(trainset)[end] / batch_size)
        
        (epoch % 5 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

        if epoch % 50 == 0

            st_ = Lux.testmode(st)

            x = model.sample(num_samples, ps, st_, rng, dev)
            
            x = x |> cpu_dev
            x = sqrt.(x[:, :, 1, :].^2 + x[:, :, 2, :].^2)   

            plot_list = []
            for i in 1:9
                push!(plot_list, heatmap(x[:, :, i])); 
            end

            save_dir = joinpath("output/train/images/", @sprintf("img_%i.pdf", epoch))
            savefig(plot(plot_list..., layout=(3,3)), save_dir)

        end

    end

end