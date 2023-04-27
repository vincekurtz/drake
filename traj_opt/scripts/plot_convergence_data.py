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
if (len(sys.argv) != 2):
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
fig, ax = plt.subplots(5,2,sharex=True,figsize=(16,11))

fig.suptitle(f"{example_name} convergence data")

#iters_tr_accepted = []
#tr_accepted = []
#eta = 0.0
#for i in range(len(iters)):
#    if data["trust_ratio"][i] > eta:
#        tr_accepted.append(data["trust_ratio"][i])
#        iters_tr_accepted.append(iters[i])
#ax[1,1].plot(iters_tr_accepted, tr_accepted)

ax[0,0].plot(iters, data["trust_ratio"])
ax[0,0].set_ylabel("trust ratio")
ax[0,0].set_ylim((-1,3))

ax[1,0].plot(iters, data["delta"], label="$\Delta$")
ax[1,0].plot(iters, data["dq_norm"], label="$\|\Delta q\|$")
ax[1,0].set_ylabel("$||\Delta q||$")
ax[1,0].set_yscale("log")
ax[1,0].legend()

ax[2,0].plot(iters, data["grad_norm"] / data["cost"])
ax[2,0].set_ylabel("$||g|| / cost$")
ax[2,0].set_yscale("log")

ax[3,0].plot(iters, data["grad_norm"])
ax[3,0].set_ylabel("$||g||$")
ax[3,0].set_yscale("log")

ax[4,0].plot(iters, -data["dL_dq"])
ax[4,0].set_ylabel(r"$\frac{|g' \Delta q|}{cost}$")
ax[4,0].set_yscale("log")

ax[0,1].plot(iters, data["merit"] - np.min(data["merit"]))
ax[0,1].set_ylabel("Merit (- baseline)")
ax[0,1].set_yscale("log")

ax[1,1].plot(iters, data["cost"] - np.min(data["cost"]))
ax[1,1].set_ylabel("Cost (- baseline)")
ax[1,1].set_yscale("log")

ax[2,1].plot(iters, data["h_norm"])
ax[2,1].set_ylabel("Constraint Viol.")
ax[2,1].set_yscale("log")

ax[3,1].plot(iters, data["ls_iters"])
ax[3,1].set_ylabel("Linesearch Iters")

ax[4,1].plot(iters, data["alpha"])
ax[4,1].set_ylabel(r"Linesearch Param $\alpha$")

ax[4,0].set_xlabel("Iteration")
ax[4,0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[4,1].set_xlabel("Iteration")
ax[4,1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
