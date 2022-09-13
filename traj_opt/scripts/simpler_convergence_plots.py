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
data_file1 = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/solver_stats.csv"
data_file2 = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/solver_stats_2.csv"
data_file3 = drake_root + f"/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/solver_stats_3.csv"

# Read data from the file and format nicely
data1 = np.genfromtxt(data_file1, delimiter=',', names=True)
data2 = np.genfromtxt(data_file2, delimiter=',', names=True)
data3 = np.genfromtxt(data_file3, delimiter=',', names=True)

# Make plots
fig, ax = plt.subplots(1,1,sharex=True,figsize=(6,6))
plt.subplots_adjust(left=0.14,
                    bottom=0.08,
                    right=0.98,
                    top=0.94,
                    wspace=0.1,
                    hspace=0.1)

fig.suptitle(f"Spinner Convergence with $F=1 N$, $\delta=1 mm$, $v_s = 0.01 m/s$")

#ax[0].plot(data1["iter"], data1["cost"])
#ax[0].plot(data2["iter"], data2["cost"])
#ax[0].plot(data3["iter"], data3["cost"])
#ax[0].set_ylabel("Cost")
#ax[0].set_yscale("log")

ax.plot(data1["iter"], -data1["dL_dq"])
#ax[1].plot(data2["iter"], -data2["dL_dq"])
#ax[1].plot(data3["iter"], -data3["dL_dq"])
ax.set_ylabel(r"Gradient ($\frac{|g' \Delta q|}{cost}$)")
ax.set_yscale("log")
ax.set_ylim(1e-16,1e1)

ax.set_xlabel("Iteration")


plt.show()
