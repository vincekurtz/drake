sim_time=10
# sim_time=1.2
mesh_res=5
# mesh_res=2
# warm up
bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --num_instances=1 --print_perf=false --simulation_sec=${sim_time} --integration_scheme=convex --visualize=false --use_error_control=false --mesh_res=${mesh_res}


num_instances_cpu=(1 10 20 50 80 100 200 500 800)
# Run all CPU
for num_instances in ${num_instances_cpu[@]}; do
   bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=false --num_instances=${num_instances} --print_perf=true --simulation_sec=${sim_time} --integration_scheme=convex --visualize=false --use_error_control=false --mesh_res=${mesh_res} 
done
