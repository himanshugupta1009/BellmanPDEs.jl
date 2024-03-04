# utils.jl


function propagate_state(state, action, Δt, veh)
    #=
    Use the dynamics function to propogate the vehicle from given state
    and given action for duration Δt
    =# 
    new_state = get_next_state(state, action, Δt, veh)
    return new_state
end

function get_next_state(s,a,Δt,veh)

    x,y,θ,v = s
    ϕ,δv = a
    L = veh.l

    new_v = v+δv
    if(new_v == 0.0)
        sp = SVector{4, Float64}(x,y,θ,new_v)
        return sp
    end
    if(ϕ == 0.0)
        new_θ = θ
        new_x = x + new_v*cos(new_θ)*Δt
        new_y = y + new_v*sin(new_θ)*Δt
    else
        new_θ = θ + (new_v * tan(ϕ) * (Δt) / L)
        new_x = x + ((L/tan(ϕ)) * (sin(new_θ) - sin(θ)))
        new_y = y + ((L/tan(ϕ)) * (cos(θ) - cos(new_θ)))
    end

    wrapped_θ = wrap_between_minus_π_and_π(new_θ)
    sp = SVector{4, Float64}(new_x,new_y,wrapped_θ,new_v)
    return sp
end

function wrap_between_minus_π_and_π(angle)
    while angle < -π    
        angle += 2π
    end
    while angle > π
        angle -= 2π
    end
    return angle
end

function interpolate_value(grid::RectangleGrid, value_array::Vector{Float64}, x::AbstractVector)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < grid.cutPoints[d][1] || x[d] > grid.cutPoints[d][end]
            val_x = -1e6
            return val_x
        end
    end
    #Interpolate value at given state
    val_x = GridInterpolations.interpolate(grid, value_array, x)
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
function multi2single_ind(grid, itr)
    state_index = 1
    for d in eachindex(itr)
        state_index += (itr[d]-1) * prod(grid.cut_counts[1:(d-1)])
    end
    return state_index
end

#=
NOTE: LazySets functions used here
=#

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

    # Check if the state is inside the goal region
    position = Singleton( SVector(x[1],x[2]) )
    if issubset(position, env.goal)
        return true
    end

    #Check if the full vehicle body is inside the goal region
    # veh_body = state_to_body_circle(x, veh)
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

    body_circle = VPolyCircle((xp_c, yp_c), veh.radius_vb)

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

function get_iterators(problem::Problem)

    (;state_range,δstate) = problem

    state_iters = [minimum(axis):δstate[i]:maximum(axis) for (i, axis) in enumerate(state_range)]

    # Gauss-Seidel sweeping scheme
    gs_iters = [[0,1] for axis in state_range]
    gs_prod = Iterators.product(gs_iters...)
    gs_list = Iterators.map(tpl -> convert(SVector{length(gs_iters), Int}, tpl), gs_prod)

    # for sweep in gs_list, need to define ind_list
    ind_gs_array = []
    for (i_gs, gs) in enumerate(gs_list)

        # for axis in sweep = [0,1,1], reverse ind_iters
        ind_iters = Array{StepRange{Int64, Int64}}(undef, size(state_range,1))
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

function get_all_states(rectangle_grid)
    num_dimensions = GridInterpolations.dimensions(rectangle_grid)
    #=
    Or you can run one of these
    1) s = rectangle_grid[1]; num_dimensions = length(s)
    2) num_dimensions = typeof(rectangle_grid).parameters[1]
    =#
    state_list = collect(map(SVector{num_dimensions,Float64},rectangle_grid))
    #=
    Or you can run this
    state_list = SVector{num_dimensions,Float64}[]
    for state in state_grid
        push!(state_list, SA[state...])
    end
    =#
    return state_list
end
