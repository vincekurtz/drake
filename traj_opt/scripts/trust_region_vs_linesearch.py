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

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()

# Bazel stores files in strange places
tr_data_file = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/solver_stats_trust_region.csv"
ls_data_file = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/solver_stats_linesearch.csv"

# Read data from the file and format nicely
tr_data = np.genfromtxt(tr_data_file, delimiter=',', names=True)
ls_data = np.genfromtxt(ls_data_file, delimiter=',', names=True)

# Get wall clock times for each
ls_wall_clock = np.cumsum(ls_data["time"])
tr_wall_clock = np.cumsum(tr_data["time"])

# Make plots
fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,6))
plt.subplots_adjust(left=0.14,
                    bottom=0.08,
                    right=0.98,
                    top=0.92,
                    wspace=0.1,
                    hspace=0.1)

fig.suptitle(f"Spinner Convergence")


ax[0].plot(tr_wall_clock, tr_data["cost"], label="trust region")
ax[0].plot(ls_wall_clock, ls_data["cost"], label="linesearch")
ax[0].set_ylabel("Cost")
ax[0].set_yscale("log")

ax[0].legend()

ax[1].plot(tr_wall_clock, -tr_data["dL_dq"])
ax[1].plot(ls_wall_clock, -ls_data["dL_dq"])
ax[1].set_ylabel(r"Gradient ($\frac{|g' \Delta q|}{cost}$)")
ax[1].set_yscale("log")
ax[1].set_ylim(1e-16,1e1)

ax[1].set_xlabel("Wall Clock Time (s)")

plt.show()
