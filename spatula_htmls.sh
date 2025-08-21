sim_time=10
mesh_res=5

num_instances=(20 50 100)

# GPU
for num_instances in ${num_instances[@]}; do
    bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=true --num_instances=${num_instances} --print_perf=false --simulation_sec=${sim_time} --integration_scheme=convex --visualize=true --use_error_control=false --mesh_res=${mesh_res}
done

# CPU
for num_instances in ${num_instances[@]}; do
    bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=false --num_instances=${num_instances} --print_perf=false --simulation_sec=${sim_time} --integration_scheme=convex --visualize=true --use_error_control=false --mesh_res=${mesh_res}
done
