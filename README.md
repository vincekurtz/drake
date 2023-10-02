# Inverse Dynamics Trajectory Optimization for Contact-Implicit Model Predictive Control

> **Note**
> This is an outdated IDTO implementation. The latest version can be found [here](https://github.com/ToyotaResearchInstitute/idto).

## Installation

Follow the standard instructions for installing [Drake](https://drake.mit.edu)
from source. 

For example, on Ubuntu:
```
git clone https://github.com/vincekurtz/drake/tree/idto_paper
cd drake
sudo ./setup/ubuntu/install_prereqs.sh
```

## Running the Examples

Once you're in the directory where this repository was cloned, just run
```
bazel run //traj_opt/examples:[example_name]
```

Example names include:
- Acrobot (no contact): `acrobot`
- Simple spinner: `spinner`
- Planar hopper: `hopper`
- Mini Cheetah quadruped: `mini_cheetah`
- 2 Jaco Arms rotate a box: `dual_jaco`
- Allegro dextrous hand rotates a sphere: `allegro_hand`

This will run a simulation which can be viewed in a web browser via Meshcat.

To enable parallel derivatives, you will need to add the `--config=omp` flag or put the
line `bulid --config=omp` in `user.bazelrc`.

## Changing the Examples

The default behavior for most examples it to run a simulation where our IDTO
optimizer is used as a controller. To change the objective or cost function,
perform open loop trajectory optimization, or explore a variety of other
options, refer to the YAML files `traj_opt/examples/[example_name].yaml`.

A few of the examples take additional gflags parameters. For example, you can
add two small hills to the Mini Cheetah example with

```
bazel run //traj_opt/examples:mini_cheetah -- --hills=2
```

