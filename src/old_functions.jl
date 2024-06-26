# planner.jl

# rng = MT(6) for left of obstacle
# rng = MT(8) for more variance on approx policy
# rng = MT(25) sends approx around top obstacle

function new_optimize_action(x::SVector{4,Float64}, Dv_RC::NTuple{K,Float64}, ia_set::SVector{L,Int},
                    actions::SVector{M,N},get_reward::Function,Dt::Float64,
                    value_array::Array{Float64,1}, veh::VehicleBody, sg) where {K,L,M,N}

    best_val = -Inf
    best_action_index = -1
    # iterate through all given action indices
    for ja in eachindex(ia_set)
        a = actions[ia_set[ja]]
        if(a[2] in Dv_RC)
            reward_x_a = get_reward(x, a, Dt, veh)
            x_p, _ = propagate_state(x, a, Dt, veh)
            val_xp = reward_x_a + interp_value(x_p, value_array, sg)
            if(val_xp > best_val)
                best_val = val_xp
                best_action_index = ja
            end
        end
    end

    return best_val, best_action_index
end
function test_new_optimize_actions(RG)
    state_k = SVector(2.0,2.0,0.0,1.0)
    actions, ia_set = RG[:f_act](state_k, RG[:Dt], RG[:veh])
    new_optimize_action(state_k, Tuple(0.5), ia_set, actions, RG[:f_cost], RG[:Dt], RG[:V], RG[:veh], RG[:sg])
end

# main function to generate a path from an initial state to goal
function plan_path(x_0, policy::Function, safe_value_lim, get_actions::Function, get_reward::Function, Dt, q_value_array, value_array, env, veh, sg, max_plan_steps)
    val_0 = interp_value(x_0, value_array, sg)

    Dv_RC_hist = rand(MersenneTwister(35), [-0.5, 0.0, 0.5], max_plan_steps)

    x_path = []
    x_subpath = []
    a_path = []
    val_path = []

    push!(x_path, x_0)
    push!(x_subpath, x_0)
    push!(val_path, val_0)

    x_k = x_0

    for plan_step in 1:max_plan_steps
        # calculate rollout action
        Dv_RC = Dv_RC_hist[plan_step]
        a_k, qval_array = policy(x_k, Dv_RC, safe_value_lim, get_actions, get_reward, Dt, q_value_array, value_array, veh, sg)

        # simulate forward one time step
        x_k1, x_k1_subpath = propagate_state(x_k, a_k, Dt, veh)

        # take value at current state (for plotting)
        val_k1 = interp_value(x_k1, value_array, sg)

        # store state and action at current time step
        push!(x_path, x_k1)
        for x_kk in x_k1_subpath
            push!(x_subpath, x_kk)
        end
        push!(a_path, a_k)
        push!(val_path, val_k1)

        # check if termination condition met
        if in_target_set(x_k1, env, veh) == true
            break
        end

        # pass state forward to next step
        x_k = deepcopy(x_k1)
    end

    return x_path, x_subpath, a_path, val_path
end

function HJB_policy(x_k, Dv_RC, safe_value_lim, get_actions::Function, get_reward::Function, Dt, q_value_array, value_array, veh, sg)
    # get actions for current state
    actions, ia_set = get_actions(x_k, Dt, veh)

    # perform optimization over full action set
    qval_x_HJB_array, val_x_HJB, ia_HJB = optimize_action(x_k, ia_set, actions, get_reward, Dt, value_array, veh, sg)

    # check for trough point
    if x_k[4] == 0.0 && actions[ia_HJB][2] == 0.0
        # take best action with Dv=+0.5
        ia_Dv_fix_set = findall(a -> a[2] > 0.0, actions)
        ia_HJB = argmax(ia -> qval_x_HJB_array[ia], ia_Dv_fix_set)

        a_ro = actions[ia_HJB]
        return a_ro, qval_x_HJB_array

    elseif x_k[4] == 0.5 && actions[ia_HJB][2] == -0.5
        # take best action with Dv=0.0 or Dv=+0.5
        ia_Dv_fix_set = findall(a -> a[2] >= 0.0, actions)
        ia_HJB = argmax(ia -> qval_x_HJB_array[ia], ia_Dv_fix_set)

        a_ro = actions[ia_HJB]
        return a_ro, qval_x_HJB_array

    else
        a_ro = actions[ia_HJB]
        return a_ro, qval_x_HJB_array
    end
end

function approx_HJB_policy(x_k, Dv_RC, safe_value_lim, get_actions::Function, get_reward::Function, Dt, q_value_array, value_array, veh, sg)
    actions, ia_set = get_actions(x_k, Dt, veh)

    # A) get near-optimal action from nearest neighbor ---
    # find nearest neighbor in state grid
    ind_s_nbrs, weights_nbrs = interpolants(sg.state_grid, x_k)
    ind_s_NN = ind_s_nbrs[findmax(weights_nbrs)[2]]

    # minimize Q-value to find best action
    qvals_NN = q_value_array[ind_s_NN]
    ia_NN_opt = argmax(ia -> qvals_NN[ia], ia_set)

    # check if ia_NN_opt is a valid action
    x_p, _ = propagate_state(x_k, actions[ia_NN_opt], Dt, veh)
    val_NN_opt = interp_value(x_p, value_array, sg)

    # println("val_NN_opt = ", val_NN_opt)

    if val_NN_opt >= safe_value_lim
        a_ro = actions[ia_NN_opt]

        return a_ro
    end

    # println("taking HJB action")

    # B) if NN action is not valid, then find pure HJB best action ---
    _, _, ia_HJB = optimize_action(x_k, ia_set, actions, get_reward, Dt, value_array, veh, sg)

    a_ro = actions[ia_HJB]

    return a_ro
end

function reactive_policy(x_k::SVector{4,Float64}, Dv_RC::Float64, safe_value_lim::Float64,
                        get_actions::Function, get_reward::Function,
                        Dt::Float64, q_value_array::Array{Array{Float64, 1}, 1},
                        value_array::Array{Float64,1}, veh::VehicleBody, sg)
    # get actions for current state
    actions, ia_set = get_actions(x_k, Dt, veh)
    # println("HG")
    # A) find best phi for Dv given by reactive controller ---
    ia_RC_set = findall(a -> a[2] == Dv_RC, actions)
    # ia_RC_set = ia_set
    # ia = SVector{length(ia_RC_set),Int}(ia_RC_set)
    # ia = SVector(ia_RC_set...)

    # total_num_actions = length(ia_set)
    # # # total_num_actions = 7
    # mask = SVector{total_num_actions,Int}(ia_set)
    # for i in 1:total_num_actions
    # # for i in eachindex(mask)
    #     if(i%2==0)
    #         # mask[i] = true
    #     else
    #         # mask[i] = false
    #     end
    # end

    qval_x_RC_array, val_x_RC, ia_RC = optimize_action(x_k, ia_RC_set, actions, get_reward, Dt, value_array, veh, sg)

    # ia_RC_set = findall(a -> a[2] == Dv_RC, actions)
    # ia = SVector{length(ia_RC_set),Int}(ia_RC_set)
    # ia = SVector(ia_RC_set...)

    # check if [Dv_RC, phi_best_RC] is a valid action in static environment ---
    if val_x_RC >= safe_value_lim
        a_ro = actions[ia_RC]

        return a_ro, qval_x_RC_array
    end

    # B) if RC action is not valid, then find pure HJB best action ---
    qval_x_HJB_array, val_x_RC, ia_HJB = optimize_action(x_k, ia_set, actions, get_reward, Dt, value_array, veh, sg)

    a_ro = actions[ia_HJB]

    return a_ro, qval_x_HJB_array
end

function test_reactive_policy(RG)
    state_k = SVector(2.0,2.0,0.0,1.0)
    delta_speed = 0.5
    safe_value_lim = 750.0
    one_time_step = 0.5
    reactive_policy(state_k,delta_speed,safe_value_lim,RG[:f_act],RG[:f_cost],one_time_step,RG[:Q],RG[:V],RG[:veh],RG[:sg])
end

function better_reactive_policy(x_k::SVector{4,Float64}, Dv_RC::Float64, safe_value_lim::Float64,
                        get_actions::Function, get_reward::Function,
                        Dt::Float64, q_value_array::Array{Array{Float64, 1}, 1},
                        value_array::Array{Float64,1}, veh::VehicleBody, sg)
    # get actions for current state
    actions, ia_set = get_actions(x_k, Dt, veh)

    velocity_set = Tuple(Dv_RC)
    val_x_RC, ia_RC = new_optimize_action(x_k, velocity_set, ia_set, actions, get_reward, Dt, value_array, veh, sg)

    # check if [Dv_RC, phi_best_RC] is a valid action in static environment ---
    if val_x_RC >= safe_value_lim
        a_ro = actions[ia_RC]
        return a_ro
    end

    # B) if RC action is not valid, then find pure HJB best action ---
    # velocity_set = (0.0,-Dv_RC)
    velocity_set = (-Dv_RC,0.0,Dv_RC)
    val_x_RC, ia_HJB = new_optimize_action(x_k, velocity_set, ia_set, actions, get_reward, Dt, value_array, veh, sg)
    a_ro = actions[ia_HJB]

    return a_ro
end
#=
RG = run_HJB(false);
=#
function test_better_reactive_policy(RG)
    state_k = SVector(2.0,2.0,0.0,1.0)
    delta_speed = 0.5
    safe_value_lim = 0.0
    one_time_step = 0.5
    better_reactive_policy(state_k,delta_speed,safe_value_lim,RG[:f_act],RG[:f_cost],one_time_step,RG[:Q],RG[:V],RG[:veh],RG[:sg])
end

# ISSUE: need to add q_val return for this function
function approx_reactive_policy(x_k, Dv_RC, safe_value_lim, get_actions::Function, get_reward::Function, Dt, q_value_array, value_array, veh, sg)
    # get actions for current state
    actions, ia_set = get_actions(x_k, Dt, veh)

    # A) get near-optimal RC action from nearest neighbor ---
    # limit action set based on reactive controllers
    ia_RC_set = findall(a -> a[2] == Dv_RC, actions)

    # find nearest neighbor in state grid
    ind_s_nbrs, weights_nbrs = interpolants(sg.state_grid, x_k)
    ind_s_NN = ind_s_nbrs[findmax(weights_nbrs)[2]]

    # minimize Q-value over RC limited action set to find best action
    qval_x_NN_array = q_value_array[ind_s_NN]
    ia_NN_RC = argmax(ia -> qval_x_NN_array[ia], ia_RC_set)

    # check if ia_NN_RC is a valid action
    x_p, _ = propagate_state(x_k, actions[ia_NN_RC], Dt, veh)
    val_NN_RC = interp_value(x_p, value_array, sg)

    # println("val_NN_RC_best = ", val_NN_RC_best)

    if val_NN_RC >= safe_value_lim
        a_ro = actions[ia_NN_RC]

        return a_ro, qval_x_NN_array
    end

    # println("taking HJB action")

    # B) if NN RC action is not valid, then find pure HJB best action ---
    qval_x_HJB_array, _, ia_HJB = optimize_action(x_k, ia_set, actions, get_reward, Dt, value_array, veh, sg)

    a_ro = actions[ia_HJB]

    return a_ro, qval_x_HJB_array
end


# approx reactive HJB_policy
#   - need to store RC best action for each possible Dv input {-, 0, +}
#   - create multi-dimensional version of ia_opt_array, should be 3 ia_RC_best stored
#   - don't want to modify optimize_actions(), just feed in different ia_sets for each Dv
#       - want to define these sets once, so don't have to keep using findall(Dv)

#   - could also just store Q(s,a) for every action at every state
#   - would do Dv_RC filtering during application, instead of having to deal with specifically during solving
#   - for each state, would have array of values same length as action set
#   - would just need to do some find/min operations on the list to find desired action indices


#   - for given state, have 2^4=16 neighboring grid nodes surrounding it
#   - each neighboring grid node has a Q-value for every action (21 actions)
#   - RC will limit the action set to a single Dv, same limited set will be considered at each neighbor (7 actions)
#       - function: given Dv_RC, output limited ia set (ia_RC_set)
#   - at each neighbor, can get ia_RC_best by minimizing Q-values over limited RC set

#   - with ia_RC_best at each neighbor, can do same set operations from old approx method (sort, unique, ...)
#       - (?): need to assemble list, or just try nearest neighbor first?
#   - will end up with short list of actions
#       - these are the ia_RC_best actions at the exact neighboring node states
#       - in other words, would be true optimal action if vehicle was at node state
#       - instead, can just assume that closest node action is near optimal for our state
#       - need to check subsequent state for collisions/RIC (value-based), but otherwise can take action

#   - (?): is value being bounded? or just checking validity
#       - bounding value is a little trickier than before, because RC constraint means none of the RC neighbors may be true optimal to Dt reward function
#       - however, minimum RC Q-value at neighboring nodes is true best case performance under RC constraint, should be able to use this if needed

#   - MAIN IDEA: instead of perfoming Q-value optimization at exact state, choose minimum from stored Q-values at neighboring nodes


function old_propagate_state(x_k, a_k, Dt, veh)
    # define number of substeps used for integration
    substeps = 10
    Dt_sub = Dt / substeps

    x_k1_subpath = MVector{substeps, SVector{4, Float64}}(undef)

    # step through substeps from x_k
    x_kk = x_k
    for kk in 1:substeps
        # Dv applied on first substep only
        kk == 1 ? a_kk = a_k : a_kk = SVector{2, Float64}(a_k[1], 0.0)

        # propagate for Dt_sub
        x_kk1 = discrete_time_EoM(x_kk, a_kk, Dt_sub, veh)

        # store new point in subpath
        x_k1_subpath[kk] = x_kk1

        # pass state to next loop
        x_kk = x_kk1
    end

    x_k1 = x_kk

    return x_k1, SVector(x_k1_subpath)
end
function discrete_time_EoM(x_k, a_k, Dt, veh)
    #=
    NOTE: THIS FUNCTION IS WRONG. I can't believe Will computed the next state incorrectly.
    No wonder the vehicle got stuck sometimes with the correct dynamics function *facepalm*.

    Will computed x_dot, y_dot, theta_dot and then approximated new state with old_state+derivative*time
    This is an approximation and causes issues. We have the exact math availablle, why approximate it then?
    
    "get_next_state" is the correct function to use here.
    =#

    # break out current state
    xp_k = x_k[1]
    yp_k = x_k[2]
    theta_k = x_k[3]
    v_k = x_k[4]

    # break out action
    phi_k = a_k[1]
    Dv_k = a_k[2]

    # calculate derivative at current state
    xp_dot_k = (v_k + Dv_k) * cos(theta_k)
    yp_dot_k = (v_k + Dv_k) * sin(theta_k)
    theta_dot_k = (v_k + Dv_k) * 1/veh.l * tan(phi_k)

    # calculate next state with linear approximation
    xp_k1 = xp_k + (xp_dot_k * Dt)
    yp_k1 = yp_k + (yp_dot_k * Dt)
    theta_k1 = theta_k + (theta_dot_k * Dt)
    v_k1 = v_k + (Dv_k)

    # modify angle to be within [-pi, +pi] range
    theta_k1 = theta_k1 % (2*pi)
    if theta_k1 > pi
        theta_k1 -= 2*pi
    elseif theta_k1 < -pi
        theta_k1 += 2*pi
    end

    # reassemble state vector
    x_k1 = SVector{4, Float64}(xp_k1, yp_k1, theta_k1, v_k1)

    return x_k1
end