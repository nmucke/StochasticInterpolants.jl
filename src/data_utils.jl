

using NPZ
using JSON
using FileIO


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