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

    # if typeof(model) == ScoreMatchingLangevinDynamics
    #     get_t_batch(batch_size) = rand(rng, Float32, (batch_size,));

    # elseif typeof(model) == DenoisingDiffusionProbabilisticModel
    #     get_t_batch(batch_size) = rand(rng, 0:model.timesteps-1, (batch_size,));
    # end

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

            t = rand(rng, Float32, (batch_size,)) |> dev #get_t_batch(batch_size) |> dev;

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
        
        (epoch % 10 == 0) && println(lazy"Loss Value after $epoch iterations: $running_loss")

        if epoch % 50 == 0

            st_ = Lux.testmode(st)
            x = model.sde_sample(num_samples, ps, st_, rng, dev)
            
            #x = model.sample(num_samples, ps, st_, rng, dev)
            # x = randn(rng, Float32, model.unet.upsample.size..., model.unet.conv_in.in_chs, num_samples)
            # x = x .* Float32.(model.marginal_probability_std(1.0)) |> dev
            # t_span = (model.eps, 1.0)

            # dt = 0.001

            # drift = Lux.Experimental.StatefulLuxLayer(model.unet, nothing, st_)

            # dudt(u, p, t) = -model.diffusion_coefficient(t).^2 .* drift((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p) 
            # g(u, p, t) = model.diffusion_coefficient(t) .* ones(size(u)) |> dev    #diffusion((u, t .* ones(1, 1, 1, size(u)[end])) |> dev, p)

            # ff = SDEFunction{false}(dudt, g)
            # prob = SDEProblem{false}(ff, g, x, reverse(t_span), ps)

            # sol = solve(
            #     prob,
            #     SRIW1(),
            #     dt = -dt,
            #     save_everystep = false,
            #     adaptive = false
            # )
            # sol = sol[:, :, :, :, end]

            
            x = x |> cpu_dev
            x = sqrt.(x[5:123, 5:123, 1, :].^2 + x[5:123, 5:123, 2, :].^2)   

            plot_list = []
            for i in 1:9
                push!(plot_list, heatmap(x[:, :, i])); 
            end

            save_dir = joinpath("output/train/images/", @sprintf("img_%i.pdf", epoch))
            savefig(plot(plot_list..., layout=(3,3)), save_dir)

        end

    end

    return ps, st

end