#Define different structs for the package

#=
Solver Parameters:
=#
struct HJBSolver{P}
    grid::RectangleGrid{P}
    Q_values::Matrix{Float64}
    V_values::Vector{Float64}
    state_type::Vector{Int}
    max_steps::Int
    ϵ::Float64 #Dval_tol in old code
    Δt::Float64
end

#=
Problem Parameters:
=#
struct Problem{N,E,V,F1<:Function,F2<:Function}
    state_range::SVector{N,Tuple{Float64,Float64}}
    δstate::SVector{N,Float64}
    env::E
    veh::V
    controls::F1
    cost::F2
end

#=
Planner Parameters:
=#
struct HJBPlanner{S,P}
    solver::S
    problem::P
    safe_value_limit::Float64
end