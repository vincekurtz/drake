#!/usr/bin/env python

##
#
# A quick script to make a plot of solution data from a trajectory optimization
# example (see example_base.h).
#
# This script must be run from the "drake/" directory. 
#
##

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import os
import sys

# Command-line flags determine which example (pendulum, acrobot, spinner) we're
# dealing with. 
possible_example_names = ["pendulum", "acrobot", "spinner"]
if (len(sys.argv) != 2) or (sys.argv[1] not in possible_example_names):
    print(f"Usage: {sys.argv[0]} {possible_example_names}")
    print("\nThe corresponding example must be run first (e.g. 'bazel run traj_opt/examples:pendulum`), with 'save_solver_stats_csv=true'")
    sys.exit(1)
example_name = sys.argv[1]

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()

# Bazel stores files in strange places
data_file = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/{example_name}.runfiles/drake/solver_stats.csv"

# Read data from the file and format nicely
data = np.genfromtxt(data_file, delimiter=',', names=True)
iters = data["iter"]

# Make plots
fig, ax = plt.subplots(5,2,sharex=True,figsize=(16,11))

fig.suptitle(f"{example_name} convergence data")

ax[0][0].plot(iters, data["cost"] - data["cost"][-1])
ax[0][0].set_ylabel("Cost (minus baseline)")
ax[0][0].set_yscale("log")

ax[1][0].plot(iters, data["time"])
ax[1][0].set_ylabel("Compute Time (s)")
ax[1][0].set_ylim((0,0.05))

ax[2][0].plot(iters, data["ls_iters"])
ax[2][0].set_ylabel("Linesearch Iters")

ax[3][0].plot(iters, data["alpha"])
ax[3][0].set_ylabel("alpha")

ax[4][0].plot(iters, data["trust_region_ratio"])
ax[4][0].set_ylabel("trust region ratio")
ax[4][0].set_ylim((-1,3))

ax[0][1].plot(iters, data["cost"])
ax[0][1].set_ylabel("Cost")
ax[0][1].set_yscale("log")

ax[1][1].plot(iters, data["grad_norm"])
ax[1][1].set_ylabel("$||g||$")
ax[1][1].set_yscale("log")

ax[2][1].plot(iters, data["dq_norm"])
ax[2][1].set_ylabel("$||\Delta q||$")
ax[2][1].set_yscale("log")

ax[3][1].plot(iters, data["grad_norm"] / data["cost"])
ax[3][1].set_ylabel("$||g|| / cost$")
ax[3][1].set_yscale("log")

ax[4][0].set_xlabel("Iteration")
ax[4][0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[4][1].set_xlabel("Iteration")
ax[4][1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
