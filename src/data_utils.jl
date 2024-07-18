

using NPZ
using JSON


function load_npz(data_path::String)
    data = npzread(data_path)
    data = data["arr_0"]
    data = permutedims(data, (3, 2, 1))
    return data    
end

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
