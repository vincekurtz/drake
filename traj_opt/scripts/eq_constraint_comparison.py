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
example_name = "hopper"
csv_names = ["no_constraint_R_1e1.csv", 
             "eq_constraint_R_1e1.csv",
             "no_constraint_R_1e3.csv",
             "eq_constraint_R_1e3.csv"]
labels = ["unconstrained R=1e1", "constrained R=1e1", "unconstrained R=1e3", "constrained R=1e3"]

# Get file locations
drake_root = os.getcwd()
data_root = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/{example_name}.runfiles/drake/"

# Make plots
fig, ax = plt.subplots(1,1,sharex=True,figsize=(8,6))
fig.suptitle(f"{example_name} convergence data")

# Get a baseline cost
N = len(csv_names)
baseline = np.inf
for i in range(N):
    data_file = data_root + csv_names[i]
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    baseline = np.min([baseline, data["cost"][-1]])

# Get the main data
for i in range(N):
    # Read data from the file and format nicely
    data_file = data_root + csv_names[i]
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    iters = data["iter"]

    #ax[0].plot(iters, data["cost"] - baseline, label=labels[i])
    #ax[0].set_ylabel("Cost (minus baseline)")
    #ax[0].set_yscale("log")

    ax.plot(iters, data["dL_dqH"], label=labels[i])
    ax.set_ylabel("Max unactuated torque")
    ax.set_yscale("log")

ax.legend()
ax.grid()
ax.set_xlabel("Iteration")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
