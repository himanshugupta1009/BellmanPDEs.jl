
JET.@report_opt ignored_modules=(Base,) test_reactive_policy(RG)

@profiler for i in 1:10000 test_reactive_policy(RG) end

state_k = SVector(2.0,2.0,0.0,1.0)
delta_speed = 0.5
safe_value_lim = 750.0
one_time_step = 0.5
f_act = RG[:f_act]
f_cost = RG[:f_cost]
Q = RG[:Q]
V = RG[:V]
veh = RG[:veh]
sg = RG[:sg]

@profiler for i in 1:10000 reactive_policy(state_k,delta_speed,safe_value_lim,f_act,f_cost,one_time_step,Q,V,veh,sg) end
@code_warntype reactive_policy(state_k,delta_speed,safe_value_lim,f_act,f_cost,one_time_step,Q,V,veh,sg)

JET.@report_opt ignored_modules=(Base,) reactive_policy(state_k,delta_speed,safe_value_lim,f_act,f_cost,one_time_step,Q,V,veh,sg)
JET.@report_opt ignored_modules=() reactive_policy(state_k,delta_speed,safe_value_lim,f_act,f_cost,one_time_step,Q,V,veh,sg)
