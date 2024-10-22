

using NPZ
using JSON
using FileIO
using YAML


function load_npz(data_path::String)
    data = npzread(data_path)
    data = data["arr_0"]
    data = permutedims(data, (3, 2, 1))
    return data    
end

"""
    load_transonic_cylinder_flow_data(
        data_folder::String,
        data_ids::Array{Int},
        state_dims::Tuple,
        num_pars::Int,
        time_step_info::Tuple
    )

Load the transonic cylinder flow data.
"""

function load_transonic_cylinder_flow_data(;
    data_folder,
    data_ids,
    state_dims,
    num_pars,
    time_step_info
)   

    start_time, num_steps, skip_steps = time_step_info

    trainset_state = zeros(state_dims...,  num_steps, length(data_ids));
    trainset_pars = zeros(num_pars, num_steps, length(data_ids));
    trajectory_counter = 1;
    for i = data_ids
        time_counter = 1;
        for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
            trajectory = lpad(string(i), 6, '0')
            timestep = lpad(string(j), 6, '0')

            velocity = load_npz("$(data_folder)/sim_$(trajectory)/velocity_$(timestep).npz")
            density = load_npz("$(data_folder)/sim_$(trajectory)/density_$(timestep).npz")
            pressure = load_npz("$(data_folder)/sim_$(trajectory)/pressure_$(timestep).npz")

            data = cat(velocity, density, pressure, dims=3)

            trainset_state[:, :, :, time_counter, trajectory_counter] = data[:, :, :]
            
            pars = JSON.parsefile("$(data_folder)/sim_$(trajectory)/src/description.json")
            trainset_pars[1, time_counter, trajectory_counter] = pars["Mach Number"]

            time_counter += 1
        end
        trajectory_counter += 1
    end

    return trainset_state, trainset_pars
end


"""
    load_incompressible_flow_data(
        data_folder::String,
        data_ids::Array{Int},
        state_dims::Tuple,
        num_pars::Int,
        time_step_info::Tuple
    )

Load the transonic cylinder flow data.
"""

function load_incompressible_flow_data(;
    data_folder,
    data_ids,
    state_dims,
    num_pars,
    time_step_info
)   

    start_time, num_steps, skip_steps = time_step_info

    trainset_state = zeros(state_dims...,  num_steps, length(data_ids));
    trainset_pars = zeros(num_pars, num_steps, length(data_ids));
    trajectory_counter = 1;
    for i = data_ids
        time_counter = 1;
        for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
            trajectory = lpad(string(i), 6, '0')
            timestep = lpad(string(j), 6, '0')

            velocity = load_npz("$(data_folder)/sim_$(trajectory)/velocity_$(timestep).npz")
            pressure = load_npz("$(data_folder)/sim_$(trajectory)/pressure_$(timestep).npz")

            data = cat(velocity, pressure, dims=3)

            trainset_state[:, :, :, time_counter, trajectory_counter] = data[:, :, :]
            
            pars = JSON.parsefile("$(data_folder)/sim_$(trajectory)/src/description.json")
            trainset_pars[1, time_counter, trajectory_counter] = pars["Reynolds Number"]

            time_counter += 1
        end
        trajectory_counter += 1
    end

    return trainset_state, trainset_pars
end


function load_turbulence_in_periodic_box_data(;
    data_folder,
    data_ids,
    state_dims,
    num_pars,
    time_step_info
) 
    start_time, num_steps, skip_steps = time_step_info

    # data_train = load("data/periodic_box_turbulence_data.jld2", "data_train");
    data_train = load(data_folder, "data_train");
    trainset_state = zeros(state_dims...,  num_steps, length(data_ids));
    for i in 1:length(data_ids)
        time_counter = 1
        for j in start_time:skip_steps:(start_time+skip_steps*num_steps-1)
            trainset_state[:, :, 1, time_counter, i] = data_train[i].data[1].u[j][1][2:end-1, 2:end-1]
            trainset_state[:, :, 2, time_counter, i] = data_train[i].data[1].u[j][2][2:end-1, 2:end-1]
            time_counter += 1
        end
    end
    
    trainset_pars = zeros(num_pars, num_steps, length(data_ids));

    return trainset_state, trainset_pars

end



function prepare_data_for_time_stepping(
    trainset,
    trainset_pars;
    len_history = 1,
)
    H, W, C, num_steps, num_trajectories = size(trainset)
    pars_dim = size(trainset_pars, 1)

    trainset_init_distribution = zeros(H, W, C, len_history, num_steps-len_history, num_trajectories);
    trainset_target_distribution = zeros(H, W, C, num_steps-len_history, num_trajectories);
    for i in 1:num_trajectories
        for step = 1:num_steps-len_history
            trainset_init_distribution[:, :, :, :, step, i] = trainset[:, :, :, step:(step+len_history-1), i];
            trainset_target_distribution[:, :, :, step, i] = trainset[:, :, :, step+len_history, i];
        end;
    end;
    trainset_pars = trainset_pars[:, 1:(num_steps-len_history), :];
    
    trainset_init_distribution = reshape(trainset_init_distribution, H, W, C, len_history, (num_steps-len_history)*num_trajectories);
    trainset_target_distribution = reshape(trainset_target_distribution, H, W, C, (num_steps-len_history)*num_trajectories);
    trainset_pars_distribution = reshape(trainset_pars, pars_dim, (num_steps-len_history)*num_trajectories);
    

    return trainset_init_distribution, trainset_target_distribution, trainset_pars_distribution
end

function load_test_case_data(
    test_case, test_args,
)

    T = Float32;

    # Load the test case configuration
    test_case_config = YAML.load_file("configs/test_cases/$test_case.yml");

    # Get dimensions of state and parameter spaces
    H = test_case_config["state_dimensions"]["height"];
    W = test_case_config["state_dimensions"]["width"];
    C = test_case_config["state_dimensions"]["channels"];
    if !isnothing(test_case_config["parameter_dimensions"])
        pars_dim = length(test_case_config["parameter_dimensions"]);
        num_pars = pars_dim
    else
        pars_dim = 1;
        num_pars = 0;
    end;

    # Get data path
    data_folder = test_case_config["data_folder"];

    # Time step information
    start_time = test_case_config["training_args"]["time_step_info"]["start_time"];
    num_steps = test_case_config["training_args"]["time_step_info"]["num_steps"];
    skip_steps = test_case_config["training_args"]["time_step_info"]["skip_steps"];

    test_start_time = test_case_config["test_args"][test_args]["time_step_info"]["start_time"];
    test_num_steps = test_case_config["test_args"][test_args]["time_step_info"]["num_steps"];
    test_skip_steps = test_case_config["test_args"][test_args]["time_step_info"]["skip_steps"];


    if test_case == "transonic_cylinder_flow"
        # Load the training data
        trainset, trainset_pars = load_transonic_cylinder_flow_data(
            data_folder=data_folder,
            data_ids=test_case_config["training_args"]["ids"],
            state_dims=(H, W, C),
            num_pars=pars_dim,
            time_step_info=(start_time, num_steps, skip_steps)
        );

        # Load the test data
        testset, testset_pars = load_transonic_cylinder_flow_data(
            data_folder=data_folder,
            data_ids=test_case_config["test_args"][test_args]["ids"],
            state_dims=(H, W, C),
            num_pars=pars_dim,
            time_step_info=(test_start_time, test_num_steps, test_skip_steps)
        );
    elseif test_case == "incompressible_flow"
        # Load the training data
        trainset, trainset_pars = load_incompressible_flow_data(
            data_folder=data_folder,
            data_ids=test_case_config["training_args"]["ids"],
            state_dims=(H, W, C),
            num_pars=pars_dim,
            time_step_info=(start_time, num_steps, skip_steps)
        );

        # Load the test data
        testset, testset_pars = load_incompressible_flow_data(
            data_folder=data_folder,
            data_ids=test_case_config["test_args"][test_args]["ids"],
            state_dims=(H, W, C),
            num_pars=pars_dim,
            time_step_info=(test_start_time, test_num_steps, test_skip_steps)
        );

    elseif test_case == "turbulence_in_periodic_box"
        # Load the training data
        trainset, trainset_pars = load_turbulence_in_periodic_box_data(
            data_folder=data_folder,
            data_ids=test_case_config["training_args"]["ids"],
            state_dims=(H, W, C),
            num_pars=pars_dim,
            time_step_info=(start_time, num_steps, skip_steps)
        );

        # Load the test data
        testset, testset_pars = load_turbulence_in_periodic_box_data(
            data_folder=data_folder,
            data_ids=test_case_config["test_args"][test_args]["ids"],
            state_dims=(H, W, C),
            num_pars=pars_dim,
            time_step_info=(test_start_time, test_num_steps, test_skip_steps)
        );
    else
        error("Invalid test case")
    end

    # Load mask if it exists
    if test_case_config["with_mask"]
        mask = npzread("$data_folder/sim_000000/obstacle_mask.npz")["arr_0"];
        mask = permutedims(mask, (2, 1));
    else
        mask = ones(H, W, C);
    end;

    trainset = convert(Array{T}, trainset);
    trainset_pars = convert(Array{T}, trainset_pars);
    testset = convert(Array{T}, testset);
    
    if test_case_config["normalize_data"]
        # Normalize the data
        normalize_data = StandardizeData(
            test_case_config["norm_mean"], 
            test_case_config["norm_std"],
        );
        trainset = normalize_data.transform(trainset);
        testset = normalize_data.transform(testset);
    
        # Normalize the parameters
        normalize_pars = NormalizePars(
            test_case_config["pars_min"], 
            test_case_config["pars_max"]
        );
        trainset_pars = normalize_pars.transform(trainset_pars);
        testset_pars = normalize_pars.transform(testset_pars);
    else
        normalize_data = nothing;
    end;

    return trainset, trainset_pars, testset, testset_pars, normalize_data, mask, num_pars
end
