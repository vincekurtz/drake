#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

##
#
# Quick script to make a plot of linesearch residuals
# that we saved from a particular iteration
#
##

drake_root = "/home/vjkurtz/builds/drake/"

# Get data from the same files
data_file = drake_root + "bazel-out/k8-opt/bin/traj_opt/examples/pendulum.runfiles/drake/linesearch_data.csv"
data = np.genfromtxt(data_file, delimiter=',', names=True)
alpha = data["alpha"]
residual = data["residual"]

plt.plot(alpha, residual)
plt.axvline(0.0, linestyle='--', color='grey', linewidth=1)
plt.axvline(1.0, linestyle='--', color='grey', linewidth=1)
plt.xlabel(r"Linesearch Parameter $\alpha$")
plt.ylabel(r"Linesearch Residual $L(q+\alpha\Delta q) - L(q)$")

plt.show()

