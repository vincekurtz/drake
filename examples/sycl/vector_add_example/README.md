# About
Example adds two vectors (10 million elements long) to produce a third vector. The directory contains two different memory management systems in SYCL
- Buffers in `vector-add-buffers.cc`
- Unified Shared Memory (USM) with `malloc_shared` in `vector-add-usm.cc`
The example shows the warm-up time is substantially greater for `malloc_shared` compared to just using `buffer`. Also, the example assumes compilation for GPU with cuda arch 89 (`--cuda-gpu-arch=sm_89`). This will only thus run on the GPUs that have cuda arch 89 (see full list [here](https://developer.nvidia.com/cuda-gpus))

# Prerequisites
- Make sure SYCL is setup by installing the oneAPI compiler package (see instructions [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-downloa[…]packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=apt))
- Make sure backend Nvidia support is set up by following instructions from [here](https://developer.codeplay.com/products/oneapi/nvidia/2025.1.1/guides/get-started-guide-nvidia)

# Build
Build using
```bash
bazel build //examples/sycl/vector_add_example:buffers
bazel build //examples/sycl/vector_add_example:usm
```

# Run
From Drake root run using
```bash
./bazel-bin/examples/sycl/vector_add_example/buffers
./bazel-bin/examples/sycl/vector_add_example/usm
```

