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

# struct VehicleBody{G<:VPolygon}
#     l::Float64
#     body_dims::SVector{2,Float64}
#     radius_vb::Float64
#     origin_to_cent::SVector{2,Float64}
#     origin_body::G
#     phi_max::Float64
#     v_max::Float64
# end

struct VehicleBody{P,Q}
    l::Float64
    body_dims::SVector{2,Float64}
    radius_vb::Float64
    origin_to_cent::SVector{2,Float64}
    origin_body::VPolygon{P,Q}
    phi_max::Float64
    v_max::Float64
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
function define_state_grid(state_space, dx_sizes)
    #Define the iterator that has all the points we want to compute the values at in all the dimensions
    N = length(state_space)
    state_iters = [minimum(axis):dx_sizes[i]:maximum(axis) for (i, axis) in enumerate(state_space)]
    #Define a RectangleGrid for interpolation
    state_grid = RectangleGrid(state_iters...)
    return state_grid
end
#=
state_space = [[0.0, 10.0], [0.0, 10.0], [-pi, pi], [0.0, 2.0]]
state_space2 = SVector( (0.0, 10.0), (0.0, 10.0), (-pi, pi), (0.0, 2.0) )
=#
