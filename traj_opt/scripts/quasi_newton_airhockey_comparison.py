#!/usr/bin/env python

##
#
# Plot a comparison of convergence with and without quasi-newton approximations.
# Data must be generated using quasi_newton_airhockey_comparison.sh first. 
#
# This script must be run from the "drake/" directory. 
#
##

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import os
import sys

# List of different stiction velocities to compare
vs_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()

# Bazel stores files in strange places
data_dir = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/airhockey.runfiles/drake/"

# Set up the plots
fig, ax = plt.subplots(2,1,sharex=True,figsize=(10,8))
plt.subplots_adjust(left=0.08, bottom=0.06, right=0.76, top=0.93)
fig.suptitle(f"Airhockey Convergence")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(len(vs_list)):
    vs = vs_list[i]

    data_file = f"{data_dir}solver_stats_vs_{vs}_quasi_newton_false.csv"
    data_file_qn = f"{data_dir}solver_stats_vs_{vs}_quasi_newton_true.csv"

    # Read data from the file and format nicely
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    data_qn = np.genfromtxt(data_file_qn, delimiter=',', names=True)
    iters = data["iter"]

    # Get minimum cost from both approaches
    baseline = np.min([data["cost"][-1], data_qn["cost"][-1]])

    ax[0].plot(iters, data["cost"] - baseline, 
            color=colors[i],
            linestyle=":",
            label=f"vs={vs}, qn=false")
    ax[0].plot(iters, data_qn["cost"] - baseline, 
            color=colors[i],
            label=f"vs={vs}, qn=true")
    ax[0].set_ylabel("Cost (minus baseline)")
    ax[0].set_yscale("log")


    ax[1].plot(iters, data["grad_norm"] / data["cost"],
            color=colors[i],
            linestyle=":",
            linewidth=1,
            label=f"vs={vs}, qn=false")
    ax[1].plot(iters, data_qn["grad_norm"] / data_qn["cost"],
            color=colors[i],
            label=f"vs={vs}, qn=true")

    ax[1].set_ylabel("$||g|| / cost$")
    ax[1].set_yscale("log")

ax[0].axvline(500, color="grey", linestyle="--")
ax[1].axvline(500, color="grey", linestyle="--")

ax[1].set_xlabel("Iteration")
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

ax[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
