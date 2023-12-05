# definitions.jl

# struct Environment{N}
#     workspace::VPolygon
#     obstacle_list::SVector{N,VPolygon}
#     goal::VPolygon
# end

struct Environment{P,Q,R}
    workspace::P
    obstacle_list::Q
    goal::R
end

struct VehicleBody
    l::Float64
    body_dims::SVector{2,Float64}
    radius_vb::Float64
    origin_to_cent::SVector{2,Float64}
    origin_body::VPolygon
    phi_max::Float64
    v_max::Float64
end

struct StateGrid{P,Q}
    state_grid::RectangleGrid
    state_list_static::P
    angle_wrap_array::Array{Bool}
    ind_gs_array::Q
end

# defines environment geometry
function define_environment(workspace, obstacle_list, goal)
    return Environment(workspace, obstacle_list, goal)
end

# defines vehicle geometry
function define_vehicle(wheelbase, body_dims, origin_to_cent, phi_max, v_max)
    radius_vb = sqrt((0.5*body_dims[1])^2 + (0.5*body_dims[2])^2)

    x0_min = origin_to_cent[1] - 1/2*body_dims[1]
    x0_max = origin_to_cent[1] + 1/2*body_dims[1]
    y0_min = origin_to_cent[2] - 1/2*body_dims[2]
    y0_max = origin_to_cent[2] + 1/2*body_dims[2]
    origin_body = VPolygon([
                        SVector(x0_min, y0_min),
                        SVector(x0_max, y0_min),
                        SVector(x0_max, y0_max),
                        SVector(x0_min, y0_max)
                        ])

    veh = VehicleBody(wheelbase, body_dims, radius_vb, origin_to_cent, origin_body, phi_max, v_max)
    return veh
end

# discretizes state space
function define_state_grid(state_space, dx_sizes, angle_wrap)
    #Define the iterator that has all the points we want to compute the values at in all the dimensions

    N = length(state_space)
    state_iters = [minimum(axis):dx_sizes[i]:maximum(axis) for (i, axis) in enumerate(state_space)]
    #Define a RectangleGrid
    state_grid = RectangleGrid(state_iters...)

    state_list_static = SVector{N,Float64}[]
    for state in state_grid
        push!(state_list_static, SA[state...])
    end

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

    return state_grid, state_list_static, angle_wrap, ind_gs_array

    sg = StateGrid(state_grid, state_list_static, angle_wrap, ind_gs_array)
    return sg
end
#=
state_space = [[0.0, 10.0], [0.0, 10.0], [-pi, pi], [0.0, 2.0]]
state_space2 = SVector( (0.0, 10.0), (0.0, 10.0), (-pi, pi), (0.0, 2.0) )

=#
