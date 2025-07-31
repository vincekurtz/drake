# About
Simple example of how a `SimpleMesh`, consisting of mesh vertices and elements would be transformed using Drake's `RigidTransform`. Example includes a CPU vs GPU comparison.

# Prerequisites
- Make sure SYCL is setup by installing the oneAPI compiler package (see instructions [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-downloa[…]packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=apt))
- Make sure backend Nvidia support is set up by following instructions from [here](https://developer.codeplay.com/products/oneapi/nvidia/2025.1.1/guides/get-started-guide-nvidia)

# Build
Build using
```bash
bazel build //examples/sycl/simple_mesh_example:simple_mesh
```

# Run
From root run using
```bash
./bazel-bin/examples/sycl/simple_mesh_example/mesh_test
```
For user arguments run

```bash
./bazel-bin/examples/sycl/simple_mesh_example/mesh_test --help
```
