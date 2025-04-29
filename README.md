# Drake

Model-Based Design and Verification for Robotics.

Please see the [Drake Documentation](https://drake.mit.edu) for more
information.

## SYCL Setup

Install apt packages (first time only):
- [`intel-oneapi-base-toolkit`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=apt)
- [`oneapi-nvidia-11.8`](https://developer.codeplay.com/apt/index.html)

Set environment variables (every time):
```bash
source /opt/intel/oneapi/setvars.sh
```

Run SYCL examples:
```bash
bazel run //examples/sycl:simple
```