#=
using StaticArrays
using GridInterpolations
using LazySets
using Random
using JLD2

=#

using BellmanPDEs
using StaticArrays
using JLD2


struct Environment{P,Q,R}
    workspace::P
    obstacle_list::Q
    goal::R
end


#Define the environment geometry
function get_HJB_environment(l,b)

    #=
    The weird thing with VPolygon is that it needs a Vector input, or else it fails.
    For instance, this will not work:

    workspace = VPolygon(
                SVector{4,Tuple{Float64,Float64}}( [ (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0,10.0) ] )
                )

    So, it seems best if input to VPolygon is a vector of SVectors while defining the polygon.
    =#

    workspace = VPolygon([ SVector(0.0, 0.0),
                           SVector(l, 0.0),
                           SVector(l, b),
                           SVector(0.0,b)]
                            )
    #Assumption - Circular obstacles with some known radius (x,y,r)
    workspace_obstacles =  SVector{4,Tuple{Float64,Float64,Float64}}([ (5.125, 4.875, 1.125),
                                        (6.5, 15.25, 1.5),(16.25, 11.0, 1.125),(10.0, 9.5, 2.25) ])
    # workspace_obstacles =  SVector{5,Tuple{Float64,Float64,Float64}}([ (6,9,2),(7,19,1.5),
    #                                     (12,25,1.5),(16,15,2.5),(26,7,1.75) ])
    # workspace_obstacles =  SVector{1,Tuple{Float64,Float64,Float64}}([(70,30,30)])

    obstacle_list = Array{VPolygon,1}()
    for obs in workspace_obstacles
        push!(obstacle_list, VPolyCircle((obs[1],obs[2]),obs[3]+0.1))
    end
    obstacle_list = SVector{length(workspace_obstacles),VPolygon{Float64, SVector{2, Float64}}}(obstacle_list)

    workspace_goal = (20.0,25.0)
    goal = VPolyCircle(workspace_goal, 1.0)

    env = Environment(workspace, obstacle_list, goal)
    return env
end


struct VehicleParameters{T}
    l::Float64
    body_dims::SVector{2,Float64}
    radius_vb::Float64
    origin_to_cent::SVector{2,Float64}
    origin_body::T
    phi_max::Float64
    delta_theta_max::Float64
    v_max::Float64
    delta_speed::Float64
end

#Define the vehicle geometry
function get_vehicle_body(body_dims,origin_to_cent)
    
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

    return origin_body
end


function get_HJB_vehicle()
    length = 0.5207
    breadth = 0.2762
    body_dims = SVector(length, breadth)
    radius_around_vehicle = sqrt((0.5*body_dims[1])^2 + (0.5*body_dims[2])^2)
    dist_origin_to_center = 0.1715
    #=
    (x,y) distance of the vehicle center to the origin of the vehicle 
    which is the center point of the rear axis
    =#
    origin_to_cent = SVector(dist_origin_to_center, 0.0) 
    vehicle_body = get_vehicle_body(body_dims,origin_to_cent)
    wheelbase = 0.324
    max_steering_angle = 0.475
    max_delta_theta = pi/4
    max_speed = 2.0
    delta_speed = 0.5

    veh_params = VehicleParameters(wheelbase, body_dims, radius_around_vehicle,
                                origin_to_cent, vehicle_body, max_steering_angle, 
                                max_delta_theta,max_speed,delta_speed)
    return veh_params
end


function HJB_actions(x, Dt, veh)
 
    (;l,phi_max,delta_theta_max,delta_speed) = veh

    v = x[4]
    vp = v + delta_speed
    vn = v - delta_speed

    phi_lim = atan(delta_theta_max * 1/Dt * 1/abs(v) * l)
    phi_lim = clamp(phi_lim, 0.0, phi_max)

    phi_lim_p = atan(delta_theta_max * 1/Dt * 1/abs(vp) * l)
    phi_lim_p = clamp(phi_lim_p, 0.0, phi_max)

    phi_lim_n = atan(delta_theta_max * 1/Dt * 1/abs(vn) * l)
    phi_lim_n = clamp(phi_lim_n, 0.0, phi_max)

    num_actions = 21
    actions = SVector{num_actions, SVector{2, Float64}}(
        (-phi_lim_n, -delta_speed),        # Dv = -Dv
        (-2/3*phi_lim_n, -delta_speed),
        (-1/3*phi_lim_n, -delta_speed),
        (0.0, -delta_speed),
        (1/3*phi_lim_n, -delta_speed),
        (2/3*phi_lim_n, -delta_speed),
        (phi_lim_n, -delta_speed),

        (-phi_lim, 0.0),       # Dv = 0.0
        (-2/3*phi_lim, 0.0),
        (-1/3*phi_lim, 0.0),
        (0.0, 0.0),
        (1/3*phi_lim, 0.0),
        (2/3*phi_lim, 0.0),
        (phi_lim, 0.0),

        (-phi_lim_p, delta_speed),        # Dv = +Dv
        (-2/3*phi_lim_p, delta_speed),
        (-1/3*phi_lim_p, delta_speed),
        (0.0, delta_speed),
        (1/3*phi_lim_p, delta_speed),
        (2/3*phi_lim_p, delta_speed),
        (phi_lim_p, delta_speed)
        )

    return actions
end


function HJB_cost(x, a, Dt, veh)
    return -Dt
end


function run_new_HJB(flag)

    Δt = 0.5
    ϵ = 0.1
    max_solve_steps = 200
    l = 30.0
    b = 30.0
    max_speed = 2.0
    state_range = SVector{4,Tuple{Float64,Float64}}([
                    (0.0,l), #Range in x
                    (0.0,b), #Range in y
                    (-pi,pi), #Range in theta
                    (0.0,max_speed) #Range in v
                    ])
    dx_sizes = SVector(0.5, 0.5, deg2rad(18.0), 0.5)
    env = get_HJB_environment(l,b)
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
        save("./src/HJB_rollout_guide.jld2",d)
        return R
    else
        s = load("./src/HJB_rollout_guide.jld2")
        R = s["rollout_guide"]
        return R
    end
end


#=
RG = run_new_HJB(true);
RG = run_new_HJB(false);
p = HJBPlanner(RG[:solver],RG[:problem],750.0);

Verify if the new solver code worked fine

for i in 1:length(RG[:V])
   if( round(R[:solver].V_values[i],digits=1) != round(RG[:V][i],digits=1))
       println(i)
   end
end

s = load("./old_src/HJB_rollout_guide.jld2");
R = s["rollout_guide"];
dig = 10
for i in 1:length(RG[:solver].V_values)
   if( round(R[:solver].V_values[i],digits=dig) != round(RG[:solver].V_values[i],digits=dig))
       println(i)
   end
end
=#
