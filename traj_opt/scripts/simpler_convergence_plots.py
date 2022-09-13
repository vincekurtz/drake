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
data_file1 = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/hopper.runfiles/drake/solver_stats_1.csv"
data_file2 = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/hopper.runfiles/drake/solver_stats_2.csv"
data_file3 = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/hopper.runfiles/drake/solver_stats_3.csv"

# Read data from the file and format nicely
data1 = np.genfromtxt(data_file1, delimiter=',', names=True)
data2 = np.genfromtxt(data_file2, delimiter=',', names=True)
data3 = np.genfromtxt(data_file3, delimiter=',', names=True)

# Make plots
fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,6))
plt.subplots_adjust(left=0.14,
                    bottom=0.08,
                    right=0.98,
                    top=0.92,
                    wspace=0.1,
                    hspace=0.1)

fig.suptitle(f"Hopper Convergence\n$F=10N$, $\delta=1cm$, $v_s = 0.1m/s$")

ax[0].plot(data1["iter"], data1["cost"], label="autodiff")
ax[0].plot(data2["iter"], data2["cost"], label="central differences")
ax[0].plot(data3["iter"], data3["cost"], label="forward differences")
ax[0].set_ylabel("Cost")
ax[0].set_yscale("log")

ax[0].legend()

ax[1].plot(data1["iter"], -data1["dL_dq"])
ax[1].plot(data2["iter"], -data2["dL_dq"])
ax[1].plot(data3["iter"], -data3["dL_dq"])
ax[1].set_ylabel(r"Gradient ($\frac{|g' \Delta q|}{cost}$)")
ax[1].set_yscale("log")
ax[1].set_ylim(1e-16,1e1)

ax[1].set_xlabel("Iteration")


plt.show()
