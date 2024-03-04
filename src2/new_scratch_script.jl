state_space = SVector{2,Tuple{Float64,Float64}}([
                (0.0,10.0), #Range in x
                (0.0,10.0), #Range in y
                ])
dx_sizes = SVector(0.5, 0.5)

#=
Don't import ProfileView. Apparaently, VS Code has its own version of it.
VSCodeServer has its own version of @profview which is imported by default. 
So, jusst call @profview on the function directly
=#

@profview test_func()