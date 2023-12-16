module BellmanPDEs

using StaticArrays
using GridInterpolations
using LazySets
using Random
using JLD2
using Plots

export VPolygon, VPolyCircle, SVector, SArray, Environment, VehicleBody, StateGrid
export define_environment, define_vehicle, define_state_grid
export solve_HJB_PDE
export plan_path, HJB_policy, approx_HJB_policy, reactive_policy, better_reactive_policy, approx_reactive_policy
export interp_value, in_target_set, in_obstacle_set, in_workspace, state_to_body, state_to_body_circle
export discrete_time_EoM, propagate_state, optimize_action, get_iterators, get_all_states
export plot_HJB_value, plot_HJB_path, plot_path_value
export interpolate

include("definitions.jl")
include("solver.jl")
include("planner.jl")
include("utils.jl")
include("plotting.jl")

end
