struct HJBPlanner{S,P}
    solver::S
    problem::P
    safe_value_limit::Float64
end

function find_best_action(planner, s, actions, velocity_set)  
    
    #=
    Find the best action out of all the actions for the current state
    where the velocity lie in the velocity_set 
    =#

    (;grid, V_values, Δt) = planner.solver
    (;veh, controls, cost) = planner.problem
    
    best_value = -Inf
    best_action_index = -1

    #=
    Iterate through all actions to find the best one with velocity in
    the given velocity_set
    =#
    for i in eachindex(actions)
        a = actions[i]
        if(a[2] in velocity_set)
            sp = propagate_state(s, a, Δt, veh)
            r = cost(s, a, Δt, veh)
            V_sp = r + interpolate_value(grid,V_values,sp)
            if(V_sp > best_value)
                best_value = V_sp
                best_action_index = i
            end
        end
    end

    return best_value, best_action_index
end
function test_find_best_action(planner)
    state = SVector(2.0,2.0,0.0,1.0)
    actions = planner.problem.controls(state, planner.solver.Δt, planner.problem.veh)
    velocity_set = Tuple(0.5)
    find_best_action(planner,state,actions,velocity_set)
end
#=
p = HJBPlanner(RG[:solver],RG[:problem],750.0);
@benchmark test_find_best_action($p)
@profview for i in 1:10000 test_find_best_action(p) end
=#

function final_HJB_policy(planner, state)
    
    
    #=
    Logic that Will used for this function
    1) If current vehicle velocity is 0.0, find the best action where dv>0.0 (dv=0.5 for our case)
    2) If current vehicle velocity is >0.0, find the best action where dv is either 0.0 or >0.0
    =#

    (;Δt) = planner.solver
    (;δstate, veh, controls) = planner.problem

    # get actions for current state
    actions = controls(state, Δt, veh)
    δv = δstate[4]

    if state[4] == 0.0
        velocity_set = Tuple(δv)
        best_value, best_action_index = find_best_action(planner,state,actions,velocity_set) 
        best_action = actions[best_action_index]
        return best_action
    elseif state[4] == abs(δv)
        velocity_set = (0.0,δv)
        best_value, best_action_index = find_best_action(planner,state,actions,velocity_set)
        best_action = actions[best_action_index]
        return best_action
    else
        velocity_set = (-δv,0.0,δv)
        best_value, best_action_index = find_best_action(planner,state,actions,velocity_set)
        best_action = actions[best_action_index]
        return best_action
    end
end
function test_final_HJB_policy(planner)
    state = SVector(2.0,2.0,0.0,1.0)
    final_HJB_policy(planner,state)
end
#=
p = HJBPlanner(RG[:solver],RG[:problem],750.0);
@benchmark test_final_HJB_policy($p)
@profview for i in 1:10000 test_final_HJB_policy(p) end
=#

function final_reactive_policy(planner,state,velocity_reactive_controller)
    
    #=
    Given the velocity chosen by the reactive controller, find the best action
    out of the actions with that velocity.
    If the reactive controller's action is not valid, then find the best action
    out of all the actions for the current state.
    =#

    (;Δt) = planner.solver
    (;δstate, veh, controls) = planner.problem
    δv = δstate[4]
    SVL = planner.safe_value_limit 

    #Get actions for current state
    actions = controls(state, Δt, veh)
    velocity_set = Tuple(velocity_reactive_controller)

    #Find the best action with reactive controller velocity
    best_value, best_action_index = find_best_action(planner,state,actions,velocity_set)

    #Check if this action is a valid action in static environment ---
    if (best_value >= SVL)
        best_action = actions[best_action_index]
        return best_action
    end

    #=
    If the action determined using reactive controller velocity action is not valid, then
    find the best action from all the actions for the current state
    =#
    velocity_set = (-δv,0.0,δv)
    best_value, best_action_index = find_best_action(planner,state,actions,velocity_set)
    best_action = actions[best_action_index]
    return best_action
end
function test_final_reactive_policy(planner)
    state = SVector(2.0,2.0,0.0,1.0)
    delta_speed = 0.5
    final_reactive_policy(planner,state,delta_speed)
end
#=
p = HJBPlanner(RG[:solver],RG[:problem],750.0);
@benchmark test_final_reactive_policy($p)
@profview for i in 1:10000 test_final_reactive_policy(p) end
=#