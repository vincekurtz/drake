#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

##
#
# Quick script to make a plot of linesearch residuals
# that we saved from a particular iteration
#
# This script must be run from the "drake/" directory. 
#
##

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()

if len(sys.argv) == 1:
    # No iteration specified, use the last iteration
    filename = "linesearch_data.csv"
elif len(sys.argv) == 2:
    # iteration specified
    filename = f"linesearch_data_{sys.argv[1]}.csv"
else:
    print("Usage: python plot_linesearch_residual.py [iteration_number]")
    sys.exit()

# This file is generated by TrajectoryOptimizer::SaveLinesearchResidual, which
# is not called by TrajectoryOptimizer::Solve by default. The only way it will
# exist is if either (1) linesearch fails or (2) you edit 
# TrajectoryOptimizer::Solve to save the linesearch residual at a particular
# iteration.
data_file = drake_root + "/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/" + filename
data = np.genfromtxt(data_file, delimiter=',', names=True)
alpha = data["alpha"]
residual = data["residual"]

plt.plot(alpha, residual)
plt.axvline(0.0, linestyle='--', color='grey', linewidth=1)
plt.axvline(1.0, linestyle='--', color='grey', linewidth=1)
plt.xlabel(r"Linesearch Parameter $\alpha$")
plt.ylabel(r"Linesearch Residual $L(q+\alpha\Delta q) - L(q)$")

plt.show()

