# utils.jl

#=
NOTE: needs to be general for:
  - solving value grid
  - planning over full action space
  - planning over smaller action space

  - val_x - best value result out of actions tested (single Float64)
  - ia_opt - action tested that produced best value (single Int)
  - qval_x_array - value produced by each of actions tested (array of Float64, length(ia_set))
=#

function optimize_action(x, ia_set, actions, get_reward::Function, Dt, value_array, veh, sg)
    qval_x_array = zeros(Float64, length(ia_set))

    # iterate through all given action indices
    for ja in eachindex(ia_set)
        a = actions[ia_set[ja]]

        reward_x_a = get_reward(x, a, Dt, veh)

        x_p, _ = propagate_state(x, a, Dt, veh)
        val_xp = interp_value(x_p, value_array, sg)

        qval_x_array[ja] = reward_x_a + val_xp
    end

    # get value
    val_x = maximum(qval_x_array)

    # get optimal action index
    ja_opt = findmax(qval_x_array)[2]
    ia_opt = ia_set[ja_opt]

    return qval_x_array, val_x, ia_opt
end
#=
RG = run_HJB(false);
=#
function test_optimize_actions(RG)
    state_k = SVector(2.0,2.0,0.0,1.0)
    actions, ia_set = RG[:f_act](state_k, RG[:Dt], RG[:veh])
    optimize_action(state_k, ia_set, actions, RG[:f_cost], RG[:Dt], RG[:V], RG[:veh], RG[:sg].state_grid)
end


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

function propagate_state(x_k, a_k, Dt, veh)
    # define number of substeps used for integration
    x_k1 = discrete_time_EoM(x_k, a_k, Dt, veh)
    return x_k1, 1
end

function discrete_time_EoM(x_k, a_k, Dt, veh)
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

function interp_value(x::AbstractVector, value_array::Vector{Float64}, grid::RectangleGrid)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < grid.cutPoints[d][1] || x[d] > grid.cutPoints[d][end]
            val_x = -1e6
            return val_x
        end
    end

    # interpolate value at given state
    val_x = GridInterpolations.interpolate(grid, value_array, x)

    return val_x
end
#=
RG = run_HJB(false);
function test_interp_value(RG)
    state_k = SVector(2.4,1.7,0.3,1.0)
    interp_value(state_k,RG[:V],RG[:sg].state_grid)
end
=#

function interp_value2(x::AbstractVector, value_array::Vector{Float64}, sg::StateGrid)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < sg.state_grid.cutPoints[d][1] || x[d] > sg.state_grid.cutPoints[d][end]
            val_x = -1e6
            return val_x
        end
    end

    # interpolate value at given state
    val_x = GridInterpolations.interpolate(sg.state_grid, value_array, x)

    return val_x
end

function interp_value_NN(x, value_array, sg)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < sg.state_grid.cutPoints[d][1] || x[d] > sg.state_grid.cutPoints[d][end]
            val_x = -1e6

            return val_x
        end
    end

    # take nearest-neighbor value
    ind_s_nbrs, weights_nbrs = interpolants(sg.state_grid, x)
    ind_s_NN = ind_s_nbrs[findmax(weights_nbrs)[2]]
    val_x = value_array[ind_s_NN]

    return val_x
end

# used for GridInterpolations.jl indexing
function multi2single_ind(ind_m, sg)
    ind_s = 1
    for d in eachindex(ind_m)
        ind_s += (ind_m[d]-1) * prod(sg.state_grid.cut_counts[1:(d-1)])
    end

    return ind_s
end

# NOTE: LazySets functions used here

# workspace checker
function in_workspace(x, env, veh)
    # veh_body = state_to_body(x, veh)
    veh_body = state_to_body_circle(x, veh)

    if issubset(veh_body, env.workspace)
        return true
    end

    return false
end

# obstacle set checker
function in_obstacle_set(x, env, veh)
    # veh_body = state_to_body(x, veh)
    veh_body = state_to_body_circle(x, veh)

    for obstacle in env.obstacle_list
        # if isempty(intersection(veh_body, obstacle)) == false || isempty(intersection(obstacle, veh_body)) == false
        #     return true
        # end

        if isdisjoint(veh_body, obstacle) == false
            return true
        end
    end

    return false
end

# target set checker
function in_target_set(x, env, veh)
    # state only inside goal region
    x_pos = Singleton(x[1:2])

    if issubset(x_pos, env.goal)
        return true
    end

    # # full body inside goal region
    # veh_body = state_to_body(x, veh)

    # if issubset(veh_body, env.goal)
    #     return true
    # end

    return false
end

# vehicle body transformation function
function state_to_body(x, veh)
    # rotate body about origin by theta
    theta = x[3]
    rot_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    body = linear_map(rot_matrix, veh.origin_body)

    # translate body from origin by [x, y]
    pos_vec = x[1:2]
    LazySets.translate!(body, pos_vec)

    return body
end

# vehicle body transformation function
function state_to_body_circle(x, veh)
    d = veh.origin_to_cent[1]

    xp_c = x[1] + d * cos(x[3])
    yp_c = x[2] + d * sin(x[3])

    body_circle = VPolyCircle([xp_c, yp_c], veh.radius_vb)

    return body_circle
end

# used to create circles as polygons in LazySets.jl
function VPolyCircle(center, radius)
    # number of points used to discretize edge of circle
    pts = 16
    # circle radius is used as midpoint radius for polygon faces (over-approximation)
    r_poly = radius/cos(pi/pts)
    theta_rng = range(0, 2*pi, length=pts+1)
    circular_polygon_vertices = [ SVector(center[1] + r_poly*cos(theta), center[2] + r_poly*sin(theta)) for theta in theta_rng]
    circular_polygon = VPolygon(circular_polygon_vertices)

    return circular_polygon
end


function get_iterators(state_space,dx_sizes)

    state_iters = [minimum(axis):dx_sizes[i]:maximum(axis) for (i, axis) in enumerate(state_space)]

    # Gauss-Seidel sweeping scheme
    gs_iters = [[0,1] for axis in state_space]
    gs_prod = Iterators.product(gs_iters...)
    gs_list = Iterators.map(tpl -> convert(SVector{length(gs_iters), Int}, tpl), gs_prod)

    # for sweep in gs_list, need to define ind_list
    ind_gs_array = []
    for (i_gs, gs) in enumerate(gs_list)

        # for axis in sweep = [0,1,1], reverse ind_iters
        ind_iters = Array{StepRange{Int64, Int64}}(undef, size(state_space,1))
        for (i_ax, ax) in enumerate(gs)
            if gs[i_ax] == 0.0
                # forward
                ind_iters[i_ax] = 1:1:size(state_iters[i_ax],1)
            else
                # reverse
                ind_iters[i_ax] = size(state_iters[i_ax],1):-1:1
            end
        end

        ind_prod = Iterators.product(ind_iters...)
        ind_list = Iterators.map(tpl -> convert(SVector{length(ind_iters), Int}, tpl), ind_prod)

        push!(ind_gs_array, ind_list)
    end

    return ind_gs_array
end
