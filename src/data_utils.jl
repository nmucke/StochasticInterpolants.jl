

using NPZ
using JSON


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


function load_isotropic_turbulence_data(;
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
            
            time_counter += 1
        end
        trajectory_counter += 1
    end

    return trainset_state, trainset_pars
end


function prepare_data(
    trainset,
    trainset_pars,
    pars_dim;
    len_history = 1,
)
    H, W, C, num_steps, num_trajectories = size(trainset)
    pars_dim = size(trainset_pars, 1)

    # Divide the training set into initial and target distributions
    trainset_init_distribution = trainset[:, :, :, 1:end-1, :];
    trainset_target_distribution = trainset[:, :, :, 2:end, :];
    trainset_pars = trainset_pars[:, 1:end-1, :];

    trainset_init_distribution = reshape(trainset_init_distribution, H, W, C, (num_steps-1)*num_trajectories);
    trainset_target_distribution = reshape(trainset_target_distribution, H, W, C, (num_steps-1)*num_trajectories);
    trainset_pars_distribution = reshape(trainset_pars, pars_dim, (num_steps-1)*num_trajectories);

    return trainset_init_distribution, trainset_target_distribution, trainset_pars_distribution
end