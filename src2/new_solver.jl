#=
Solver Parameters:
=#
struct HJBSolver{P}
    grid::RectangleGrid{P}
    Q_values::Matrix{Float64}
    V_values::Vector{Float64}
    state_type::Vector{Int}
    max_steps::Int
    ϵ::Float64 #Dval_tol in old code
    Δt::Float64
end

#=
Problem Parameters:
=#
struct Problem{N,A,B,C,P,Q,F1<:Function,F2<:Function}
    state_range::SVector{N,Tuple{Float64,Float64}}
    δstate::SVector{N,Float64}
    env::Environment{A,B,C}
    veh::VehicleBody{P,Q}
    controls::F1
    cost::F2
end


function HJBSolver(problem::Problem;max_steps,ϵ,Δt)

    (;state_range,δstate) = problem
    state_iters = [minimum(axis):δstate[i]:maximum(axis) for (i, axis) in enumerate(state_range)]
    #Define a RectangleGrid for interpolation
    grid = RectangleGrid(state_iters...)

    # initialize data arrays
    println("Initializing Data Arrays ---")
    Q_values, V_values, state_type = initialize_value_arrays(problem,grid)

    return HJBSolver(grid,
                    Q_values,
                    V_values,
                    state_type,
                    max_steps,
                    ϵ,
                    Δt
                    )
end

# main function to iteratively calculate HJB value function
function solve(solver::HJBSolver, problem::Problem)

    (;grid, Q_values, V_values, state_type, max_steps, ϵ, Δt) = solver
    (;env, veh, controls, cost) = problem
    num_GS_sweeps = 2^dimensions(grid)
    GS_step = 1
    num_solve_steps = 1
    max_ΔV = Inf
    discretized_states = get_all_states(grid)
    GS_iterators = get_iterators(problem)

    println("Solving the HJB PDE now ---")

    #Main loop
    while num_solve_steps < max_steps && max_ΔV >= ϵ

        max_ΔV = -Inf
        current_GS_iterator = GS_iterators[GS_step]

        for index in current_GS_iterator
            state_index = multi2single_ind(grid, index)     # SPEED: able to speed up? haven't looked
            # If the state is in free space, update its values
            if state_type[state_index] == 2
                state = discretized_states[state_index]
                old_V = V_values[state_index]   # Store the old V value
                update_state_values(state, state_index, solver, problem) # Modify Q and V values
                new_V = V_values[state_index]   # Find the new V value
                ΔV = abs(new_V-old_V)
                max_ΔV = (ΔV > max_ΔV) ? ΔV : max_ΔV
            end
        end

        println("Solve Step: ", num_solve_steps, ", GS step: ", GS_step, ", Infinity norm of |Vᴷ - Vᴷ⁻¹| = ", max_ΔV)

        #Update counters
        GS_step = (GS_step == num_GS_sweeps) ? 1 : GS_step+1
        num_solve_steps += 1
    end
end

#=
Q: Will had this in his comments for his function? Why did he say that?
"ISSUE: seems like q_value_array is not being updated properly"
=#
function update_state_values(curr_state, state_index, solver, problem)

    (;grid, Q_values, V_values, max_steps, ϵ, Δt) = solver
    (;veh,controls,cost) = problem

    actions, _ = controls(curr_state, Δt, veh)
    max_Q_value = -Inf

    for i in 1:length(actions)
        a = actions[i]
        immediate_cost = cost(curr_state, a, Δt, veh)
        next_state, _ = propagate_state(curr_state, a, Δt, veh)
        V_next_state = interp_value(grid, V_values, next_state)
        new_Q_value = immediate_cost + V_next_state
        Q_values[state_index,i] = new_Q_value
        max_Q_value = (new_Q_value>max_Q_value) ? new_Q_value : max_Q_value
    end
    V_values[state_index] = max_Q_value
end

# initialize arrays
# function initialize_value_arrays(Dt, get_actions::Function, sg, env, veh, discretized_states, ind_gs_array)
function initialize_value_arrays(problem::Problem, grid)

    (;env,veh,controls) = problem
    state = grid[1]
    actions,_ = controls(state, 0.5, veh)

    N_states = length(grid)
    N_actions = length(actions)
    Q_values = Matrix{Float64}(undef, N_states, N_actions)
    V_values = Vector{Float64}(undef, N_states)
    state_type = Vector{Int}(undef, N_states)
    discretized_states = get_all_states(grid)

    target_set_value = 1000.0
    target_set_Q_values = target_set_value*ones(N_actions)

    default_value = -1e6
    default_Q_values = default_value*ones(N_actions)

    unsafe_set_value = -1e6
    unsafe_set_Q_values = unsafe_set_value*ones(N_actions)

    println("Initializing Value Function Arrays")
    println("Total Grid Nodes = ", N_states)

    num_grids = 1
    for state_index in 1:N_states
        state = discretized_states[state_index]

        if ( !in_workspace(state, env, veh) || in_obstacle_set(state, env, veh) )
            Q_values[state_index,:] = unsafe_set_Q_values
            V_values[state_index] = unsafe_set_value
            state_type[state_index] = 0    #Denotes this point in space is invalid
        elseif in_target_set(state, env, veh)
            Q_values[state_index,:] = target_set_Q_values
            V_values[state_index] = target_set_value
            state_type[state_index] = 1    #Denotes this point in space is valid and is in the target
        else
            Q_values[state_index,:] = default_Q_values
            V_values[state_index] = default_value
            state_type[state_index] = 2    #Denotes this point in space is valid and is free space
        end

        if num_grids % 1000 == 0
            println("Just finished initializing grid number: ", num_grids)
        end

        num_grids += 1
    end

    return Q_values, V_values, state_type
end
