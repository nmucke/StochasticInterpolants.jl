using NPZ
using JLD2
using Statistics


model_names = [
    "acdm-r10",
    "acdm-r25",
    "acdm-r50",
];
model_save_names = [
    "acdm_",
    "acdm_",
    "acdm_",
];

for (model_name, model_save_name) in zip(model_names, model_save_names)
    println("Loading $model_name")

    num_generator_steps = parse(Int, model_name[end-1:end])

    data = npzread("results/$model_name.npz")["arr_0"][:, :, 1:2, :, :, :]
    println("Mean of data: ", mean(data))

    println("Saving $model_save_name$(num_generator_steps).jld2")
    save("results/$model_save_name$(num_generator_steps).jld2", "data", data)
end

data = npzread("results/refiner-r4_std0.000001.npz")["arr_0"][:, :, 1:2, :, :, :];
save("results/refiner_4.jld2", "data", data)
    


