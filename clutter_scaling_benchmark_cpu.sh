# warm up run - CPU
bazel run //examples/multibody/clutter:clutter --cxxopt=-Wno-maybe-uninitialized --cxxopt=-Wno-uninitialized --cxxopt=-Wno-stringop-overflow -- --use_sycl=false --objects_per_pile=5 --print_perf=true --visualize=false --box_resolution=1.0 --sphere_resolution=0.02

OBJECT_PER_PILE_COUNTS=(1 2 5 10 20 33 50)
SPHERE_RESOLUTION_COUNTS=(0.005 0.01 0.02 0.04)


# CPU runs
for objects_per_pile in "${OBJECT_PER_PILE_COUNTS[@]}"; do
  for sphere_resolution in "${SPHERE_RESOLUTION_COUNTS[@]}"; do
    bazel run //examples/multibody/clutter:clutter --cxxopt=-Wno-maybe-uninitialized --cxxopt=-Wno-uninitialized --cxxopt=-Wno-stringop-overflow -- --use_sycl=false --objects_per_pile=$objects_per_pile --print_perf=true --visualize=false --box_resolution=1.0 --sphere_resolution=$sphere_resolution
  done
done