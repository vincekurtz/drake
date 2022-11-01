#!/usr/bin/env python

##
#
# Compare convergence with various geodesic acceleration settings
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
data_file_root = drake_root + "/bazel-out/k8-opt/bin/traj_opt/examples/airhockey.runfiles/drake/"

# Make plots
fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,9))

fig.suptitle("airhockey example, vs=0.01, delta=0.01")

# Read data from the files and format nicely
scenarios = ["accel_no_uphill_no", "accel_yes_uphill_all", "accel_yes_uphill_some", "accel_yes_uphill_no"]
for scenario in scenarios:
    data = np.genfromtxt(data_file_root + f"solver_stats_{scenario}.csv", delimiter=",", names=True)

    iters = data["iter"]

    ax[0].plot(iters, data["cost"], label=scenario)
    ax[0].set_ylabel("Cost")
    ax[0].set_yscale("log")

    ax[1].plot(iters, data["grad_norm"], label=scenario)
    ax[1].set_ylabel("$||g||$")
    ax[1].set_yscale("log")

ax[1].set_xlabel("Iteration")
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()

plt.show()
