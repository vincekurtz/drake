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
data_file = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/solver_stats.csv"

# Read data from the file and format nicely
data = np.genfromtxt(data_file, delimiter=',', names=True)
iters = data["iter"]

# Make plots
fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,6))

#fig.suptitle(f"spinner convergence data")

ax[0].plot(iters, data["cost"])
ax[0].set_ylabel("Cost")
ax[0].set_yscale("log")

ax[1].plot(iters, -data["dL_dq"])
ax[1].set_ylabel(r"Gradient ($\frac{|g' \Delta q|}{cost}$)")
ax[1].set_yscale("log")

ax[1].set_xlabel("Iteration")


plt.show()
