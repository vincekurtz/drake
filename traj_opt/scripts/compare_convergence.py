#!/usr/bin/env python

##
#
# Quick script to compare basic convergence data from several csv log files.
# These log files should be manually copied from solver_stats.csv first. 
#
# This script must be run from the "drake/" directory. 
#
##

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import os

# Basic parameters: set these to define the location and name of the log files
# that we'll compare, as well as corresponding legend labels
example_name = "airhockey"
csv_names = ["solver_stats_scaled.csv", "solver_stats_unscaled.csv"]
labels = ["scaled", "not scaled"]

# Get file locations
drake_root = os.getcwd()
data_root = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/{example_name}.runfiles/drake/"

# Make plots
fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6))
fig.suptitle(f"{example_name} convergence data")

# Iterate over csv files
N = len(csv_names)
for i in range(N):
    # Read data from the file and format nicely
    data_file = data_root + csv_names[i]
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    iters = data["iter"]

    ax[0].plot(iters, data["cost"] - data["cost"][-1], label=labels[i])
    ax[0].set_ylabel("Cost (minus baseline)")
    ax[0].set_yscale("log")

    ax[1].plot(iters, data["grad_norm"]/data["cost"], label=labels[i])
    ax[1].set_ylabel("$||g||/cost$")
    ax[1].set_yscale("log")

ax[0].legend()
ax[0].grid()
ax[1].grid()
ax[1].set_xlabel("Iteration")
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
