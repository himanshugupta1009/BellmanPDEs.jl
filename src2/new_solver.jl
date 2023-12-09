# solver.jl

#=
Solver Parameters:

=#

#=
Problem Parameters:

=#

struct HJBSolver{P}
    grid::RectangleGrid{P}
    Q_values::Matrix{Float64}
    V_values::Vector{Float64}
    max_steps::Int
    dV::Float64 #Dval_tol in old code
    dt::Float64
end

struct Problem{A,B,C,P,Q,F1<:Function,F2<:Function}
    env::Environment{A,B,C}
    veh::VehicleBody{P,Q}
    controls::F1
    cost::F2
end

# main function to iteratively calculate HJB value function
function solve_HJB_PDE(get_actions::Function, get_reward::Function, Dt, env, veh, sg,
                        state_list_static, ind_gs_array, Dval_tol, max_solve_steps)
    # initialize data arrays
    println("initializing ---")
    q_value_array, value_array, set_array = initialize_value_array(Dt, get_actions, sg, env, veh,
                                                            state_list_static, ind_gs_array)

    num_gs_sweeps = 2^dimensions(sg.state_grid)

    # main function loop
    println("solving ---")
    gs_step = 1

    for solve_step in 1:1
        Dval_max = 0.0

        for ind_m in ind_gs_array[gs_step]
            ind_s = multi2single_ind(ind_m, sg)     # SPEED: able to speed up? haven't looked
            println(ind_s)
            # if the node is in free space, update its value
            if set_array[ind_s] == 2
                x = state_list_static[ind_s]

                # store previous value
                v_kn1 = value_array[ind_s]

                # calculate new value
                q_value_array[ind_s], value_array[ind_s], _ = update_node_value(x, get_actions, get_reward, Dt, value_array, veh, sg)

                # compare old and new values, update largest change in value
                v_k = value_array[ind_s]
                Dval = abs(v_k - v_kn1)

                if Dval > Dval_max
                    Dval_max = Dval
                end
            end
        end

        println("solve_step: ", solve_step, ", gs_step: ", gs_step, ", Dval_max = ", Dval_max)

        # check if termination condition met
        if Dval_max <= Dval_tol
            break
        end

        # update Gauss-Seidel counter
        if gs_step == num_gs_sweeps
            gs_step = 1
        else
            gs_step += 1
        end
    end

    return q_value_array, value_array
end

# ISSUE: seems like q_value_array is not being updated properly
function update_node_value(x, get_actions::Function, get_reward::Function, Dt, value_array, veh, sg)
    # using entire action set
    actions, ia_set = get_actions(x, Dt, veh)

    # find optimal action and value at state
    qval_x_array, val_x, ia_opt = optimize_action(x, ia_set, actions, get_reward::Function, Dt, value_array, veh, sg)

    return qval_x_array, val_x, ia_opt
end

# initialize arrays
# function initialize_value_arrays(Dt, get_actions::Function, sg, env, veh, discretized_states, ind_gs_array)
function initialize_value_arrays(grid, problem, discretized_states, iterators, dt)

    state = discretized_states[1]
    env = problem.env
    veh = problem.veh
    actions = problem.controls(state, dt, veh)
    # grid = solver.grid

    N_states = length(grid)
    N_actions = length(actions)
    Q_values = Matrix{Float64}(undef, N_states, N_actions)
    V_values = Vector{Float64}(undef, N_states)
    set_array = Vector{Int}(undef, N_states)

    target_set_value = 1000.0
    target_set_Q_values = target_set_value*ones(N_actions)

    default_value = -1e6
    default_Q_values = default_value*ones(N_actions)

    unsafe_set_value = -1e6
    unsafe_set_Q_values = unsafe_set_value*ones(N_actions)

    println("Initializing Value Function Array")
    println("Total Grid Nodes = ", length(grid))

    init_step = 1
    for ind_m in iterators[1]
        ind_s = multi2single_ind(ind_m, grid)
        state = discretized_states[ind_s]

        if ( !in_workspace(state, env, veh) || in_obstacle_set(state, env, veh) )
            Q_values[ind_s,:] = unsafe_set_Q_values
            V_values[ind_s] = unsafe_set_value
            set_array[ind_s] = 0
        elseif in_target_set(state, env, veh)
            Q_values[ind_s,:] = target_set_Q_values
            V_values[ind_s] = target_set_value
            set_array[ind_s] = 1
        else
            Q_values[ind_s,:] = default_Q_values
            V_values[ind_s] = default_value
            set_array[ind_s] = 2
        end

        if init_step % 1000 == 0
            println("Initializing Grid Index Number: ", init_step)
        end

        init_step += 1
    end

    return Q_values, V_values, set_array
end
