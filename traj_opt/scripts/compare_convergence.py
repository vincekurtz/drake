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

plt.rcParams.update({'font.size': 16})

# Basic parameters: set these to define the location and name of the log files
# that we'll compare, as well as corresponding legend labels
example_name = "allegro_hand"
csv_names = ["solver_stats_no_scale.csv",
             "solver_stats_scale.csv"]
labels = ["no scaling",
          "with scaling"]

# Get file locations
drake_root = os.getcwd()
data_root = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/{example_name}.runfiles/drake/"

# Make plots
fig, ax = plt.subplots(1,1,sharex=True,figsize=(8,6))
#fig.suptitle(f"{example_name} convergence data")

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

    ax.plot(iters, data["merit"], label=labels[i], linewidth=3)
    ax.set_ylabel("Merit")
    ax.set_yscale("log")
    
    #ax[1].plot(iters, data["h_norm"], label=labels[i])
    #ax[1].set_ylabel("Constraint violation $||h(q_k)||$")
    #ax[1].set_yscale("log")
    #
    #ax[2].plot(iters, data["grad_norm"]/data["cost"], label=labels[i])
    #ax[2].set_ylabel("Gradient norm $||g||/L(q)$")
    #ax[2].set_yscale("log")

ax.legend()
ax.grid()
#ax[1].grid()
#ax[2].grid()
ax.set_xlabel("Iteration")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
