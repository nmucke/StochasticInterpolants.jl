using BSON
using YAML
using DelimitedFiles


"""
    create_checkpoint_directory(
        test_problem::String,
        model_name::String;
        base_folder::String = "trained_models"
    )
"""
function create_checkpoint_directory(
    test_problem::String,
    model_name::String;
    base_folder::String = "trained_models"
)
    base_dir = joinpath(base_folder, test_problem, model_name)
    if !isdir(base_dir)
        mkdir(base_dir)
    end

    figures_dir = joinpath(base_dir, "figures")
    if !isdir(figures_dir)
        mkdir(figures_dir)
    end

    return base_dir, figures_dir
end

"""
    save_checkpoint(ps, st, opt_st, output_dir, epoch)
"""
function save_checkpoint(;
    ps::NamedTuple, 
    st::NamedTuple,
    opt_st::NamedTuple,
    output_dir::String,
    checkpoint_name::String = "checkpoint"
)

    # If path does not exist, create it
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    path = joinpath(output_dir, "$checkpoint_name.bson")

    return bson(
        path,
        Dict(
            :ps => cpu(ps),
            :st => cpu(st),
            :opt_st => cpu(opt_st)
        )
    )
end

function matrix_to_dictionary(matrix)
    dict = Dict{Int, Float32}()
    for i in 1:size(matrix, 1)
        key = parse(Int, matrix[i, 1])
        value = parse(Float32, matrix[i, 2])
        dict[key] = value
    end
    return dict
end

"""
    load_model_weights(path::String)
"""
function load_model_weights(
    model_path::String
)

    model_weights_and_state = BSON.load("$model_path/model.bson")

    return (
        ps=model_weights_and_state[:ps], 
        st=model_weights_and_state[:st], 
        opt_st=model_weights_and_state[:opt_st]
    )
end

"""
    load_model_config(path::String)
"""
function load_model_config(
    model_path::String
)

    config = YAML.load_file("$model_path/config.yml")

    return config
end

function load_train_tracking_data(
    model_path::String,
    delim::Char = ':'
)
    training_data_mat = readdlm("$model_path/training_loss.txt", delim, Any, '\n')

    return matrix_to_dictionary(training_data_mat)
end

function load_test_tracking_data(
    model_path::String,
    delim::Char = ':'
)

    pathwise_MSE_mat = readdlm("$model_path/test_pathwise_MSE.txt", delim, Any, '\n')
    mean_MSE_mat = readdlm("$model_path/test_mean_MSE.txt", delim, Any, '\n')

    pathwise_MSE = matrix_to_dictionary(pathwise_MSE_mat)
    mean_MSE = matrix_to_dictionary(mean_MSE_mat)

    return (; pathwise_MSE, mean_MSE)
end


"""
    CheckpointManager(
        test_problem::String,
        model_name::String,
        config::Dict;
        base_folder::String = "trained_models"
    )
"""
function CheckpointManager(
    test_problem::String,
    model_name::String;
    neural_network_config::Dict = nothing,
    data_config::Dict = nothing,
    base_folder::String = "trained_models"
)
    base_dir, figures_dir = create_checkpoint_directory(
        test_problem, 
        model_name; 
        base_folder=base_folder
    )

    if isnothing(neural_network_config)
        neural_network_config = YAML.load_file("$base_dir/neural_network_config.yml");
    else
        YAML.write_file("$base_dir/neural_network_config.yml", neural_network_config)
    end

    if isnothing(data_config)
        data_config = YAML.load_file("$base_dir/data_config.yml");
    else
        YAML.write_file("$base_dir/data_config.yml", data_config)
    end

    save_model = (ps, st, opt_st, checkpoint_name="model") -> save_checkpoint(
        ps=ps, st=st, opt_st=opt_st, output_dir=base_dir, checkpoint_name=checkpoint_name
    )

    save_figure = (fig, filename) -> savefig(joinpath(figures_dir, filename))

    write_array_to_txt_file(array, filename) = begin

        # delete file if it exists
        if isfile(joinpath(base_dir, filename))
            rm(joinpath(base_dir, filename))
        end

        open(joinpath(base_dir, filename), "w") do io
            for i in 1:length(array)
                println(io, array[i])
            end
        end
    end 

    write_dict_to_txt_file(dict, filename) = begin

        # delete file if it exists
        if isfile(joinpath(base_dir, filename))
            rm(joinpath(base_dir, filename))
        end

        open(joinpath(base_dir, filename), "w") do io
            for (key, value) in dict
                println(io, "$key: $value")
            end
        end
    end

    get_training_data = () -> load_train_tracking_data(base_dir)
    get_test_data = () -> load_test_tracking_data(base_dir)

    load_model(model_name="model") = load_model_weights("$base_dir/$model_name.bson")

    return (;
        base_dir,
        figures_dir,
        save_model,
        save_figure,
        neural_network_config,
        data_config,
        write_array_to_txt_file,
        write_dict_to_txt_file,
        load_model,
        get_training_data,
        get_test_data
    )
    
end