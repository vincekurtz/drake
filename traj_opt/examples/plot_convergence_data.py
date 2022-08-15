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
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,figsize=(8,11))
ax1.set_title(f"{example_name} convergence data")

ax1.plot(iters, data["time"])
ax1.set_ylabel("Time (s)")
ax1.set_ylim((0,0.05))

ax2.plot(iters, data["cost"] - data["cost"][-1])
ax2.set_ylabel("Cost")
ax2.set_yscale("log")

ax3.plot(iters, data["ls_iters"])
ax3.set_ylabel("Linesearch Iters")

ax4.plot(iters, data["alpha"])
ax4.set_ylabel("alpha")

ax5.plot(iters, data["grad_norm"] / data["cost"])
ax5.set_ylabel("$||g||$ / cost")
ax5.set_yscale("log")

ax5.set_xlabel("Iteration")
ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

# DEBUG
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,figsize=(8,11))
ax1.set_title(f"{example_name} extra convergence data")
ax1.plot(iters, data["grad_norm"])
ax1.set_ylabel("||g||")
ax1.set_yscale("log")

ax2.plot(iters, data["dq_norm"])
ax2.set_ylabel("|| dq ||")
ax2.set_yscale("log")

ax3.plot(iters, data["g_norm_scaled"])
ax3.set_ylabel("|| g / diag(H)||")
ax3.set_yscale("log")

ax4.plot(iters, data["trust_region_ratio"])
ax4.set_ylabel("trust region ratio")
ax4.set_ylim((0,2))

ax5.plot(iters, data["H_diag_norm"])
ax5.set_ylabel("|| diag(H)||")
ax5.set_yscale("log")

ax5.set_xlabel("Iteration")
ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
