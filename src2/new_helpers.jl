using StaticArrays
using GridInterpolations
using LazySets
using Random
using JLD2

include("new_definitions.jl")
include("new_planner.jl")
include("new_solver.jl")
include("new_utils.jl")

#=
The weird thing with VPolygon is that it needs a Vector input, or else it fails.
For instance, this will not work:

workspace = VPolygon(
                    SVector{4,Tuple{Float64,Float64}}( [ (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0,10.0) ] )
                    )

So, it seems best if input to VPolygon is a vector of SVectors while defining the polygon.
=#
function get_HJB_environment()
    l = 20.0
    b = 20.0
    workspace = VPolygon([ SVector(0.0, 0.0),
                           SVector(l, 0.0),
                           SVector(l, b),
                           SVector(0.0,b)]
                            )
    #Assumption - Circular obstacles with some known radius (x,y,r)
    workspace_obstacles =  SVector{4,Tuple{Float64,Float64,Float64}}([ (5.125, 4.875, 1.125),(6.5, 15.25, 1.5),(16.25, 11.0, 1.125),(10.0, 9.5, 2.25) ])

    obstacle_list = Array{VPolygon,1}()
    for obs in workspace_obstacles
        push!(obstacle_list, VPolyCircle((obs[1],obs[2]),obs[3]+0.1))
    end
    obstacle_list = SVector{length(workspace_obstacles),VPolygon{Float64, SVector{2, Float64}}}(obstacle_list)

    workspace_goal = (13.5,19.0)
    goal = VPolyCircle(workspace_goal, 1.0)

    env = Environment(workspace, obstacle_list, goal)
    return env
end

function get_HJB_vehicle()
    length = 1.0
    breadth = 0.5
    body_dims = SVector(length, breadth)
    dist_origin_to_center = 0.375
    origin_to_cent = SVector(dist_origin_to_center, 0.0) # (x,y) distance of the vehicle center to the origin of the vehicle which is the center point of the rear axis
    wheelbase = 0.75
    max_steering_angle = 0.475
    max_speed = 2.0
    veh_body = define_vehicle(wheelbase, body_dims, origin_to_cent, max_steering_angle, max_speed)
    return veh_body
end

function HJB_actions(x, Dt, veh)
    # set change in velocity (Dv) limit
    Dv_lim = 0.5
    # set steering angle (phi) limit
    phi_max = 0.475
    Dtheta_lim = deg2rad(45)

    v = x[4]
    vp = v + Dv_lim
    vn = v - Dv_lim

    phi_lim = atan(Dtheta_lim * 1/Dt * 1/abs(v) * veh.l)
    phi_lim = clamp(phi_lim, 0.0, phi_max)

    phi_lim_p = atan(Dtheta_lim * 1/Dt * 1/abs(vp) * veh.l)
    phi_lim_p = clamp(phi_lim_p, 0.0, phi_max)

    phi_lim_n = atan(Dtheta_lim * 1/Dt * 1/abs(vn) * veh.l)
    phi_lim_n = clamp(phi_lim_n, 0.0, phi_max)

    num_actions = 21
    actions = SVector{num_actions, SVector{2, Float64}}(
        (-phi_lim_n, -Dv_lim),        # Dv = -Dv
        (-2/3*phi_lim_n, -Dv_lim),
        (-1/3*phi_lim_n, -Dv_lim),
        (0.0, -Dv_lim),
        (1/3*phi_lim_n, -Dv_lim),
        (2/3*phi_lim_n, -Dv_lim),
        (phi_lim_n, -Dv_lim),

        (-phi_lim, 0.0),       # Dv = 0.0
        (-2/3*phi_lim, 0.0),
        (-1/3*phi_lim, 0.0),
        (0.0, 0.0),
        (1/3*phi_lim, 0.0),
        (2/3*phi_lim, 0.0),
        (phi_lim, 0.0),

        (-phi_lim_p, Dv_lim),        # Dv = +Dv
        (-2/3*phi_lim_p, Dv_lim),
        (-1/3*phi_lim_p, Dv_lim),
        (0.0, Dv_lim),
        (1/3*phi_lim_p, Dv_lim),
        (2/3*phi_lim_p, Dv_lim),
        (phi_lim_p, Dv_lim)
        )

    # ia_set = collect(1:length(actions))
    ia_set = SVector{num_actions,Int}(1:num_actions)

    return actions,ia_set
end
#=
function test_dynamic_dispatch(v::VehicleBody,func::Function)
    s = SVector(2.0,2.0,0.0,1.0)
    dt = 0.5
    a,i = func(s,dt,v)
end
=#

function HJB_cost(x, a, Dt, veh)
    return -Dt
end

function run_new_HJB(flag)

    Δt = 0.5
    ϵ = 0.1
    max_solve_steps = 200
    l = 20.0
    b = 20.0
    max_speed = 2.0
    state_range = SVector{4,Tuple{Float64,Float64}}([
                    (0.0,l), #Range in x
                    (0.0,b), #Range in y
                    (-pi,pi), #Range in theta
                    (0.0,max_speed) #Range in v
                    ])
    dx_sizes = SVector(0.5, 0.5, deg2rad(18.0), 0.5)
    env = get_HJB_environment()
    veh = get_HJB_vehicle()

    P = Problem(state_range,dx_sizes,env,veh,HJB_actions,HJB_cost)

    if(flag)
        h = HJBSolver(P,
                    max_steps=max_solve_steps,
                    ϵ=ϵ,
                    Δt=Δt
                    )
        solve(h,P)
        R = (solver = h,
            problem = P
            )
        d = Dict("rollout_guide"=>R)
        save("./src2/HJB_rollout_guide.jld2",d)
        return R
    else
        s = load("./src2/HJB_rollout_guide.jld2")
        R = s["rollout_guide"]
        return R
    end
end

#=
RG = run_new_HJB(true);
RG = run_new_HJB(false);
=#
