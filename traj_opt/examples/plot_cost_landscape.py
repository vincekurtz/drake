#!/usr/bin/env python

##
#
# Make a plot of the cost landscape for a super simple system with two DoFs and
# one timestep (2 total decision variables).
#
##

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

# Determine what exactly to plot based on command-line arguments
if len(sys.argv) == 1:
    # If no iteration number is specified, we'll use the first iteration
    i = 0
elif len(sys.argv) == 2:
    i = int(sys.argv[1])
else:
    print(f"Usage: {sys.argv[0]} [iteration = 0]")
    sys.exit(1)

# Get data for the contour plot
# N.B. Must run this script from the drake/ directory (containing bazel-out symlink)
drake_root = os.getcwd();
data_file = drake_root + "/bazel-out/k8-opt/bin/traj_opt/examples/2dof_spinner.runfiles/drake/contour_data.csv"

data = np.genfromtxt(data_file, delimiter=',', names=True)
q1 = data["q1"]
q2 = data["q2"]
cost = data["L"]

# Get iteration data (q_k, L_k, Delta_k, g_k, h_k)
data_file = drake_root + "/bazel-out/k8-opt/bin/traj_opt/examples/2dof_spinner.runfiles/drake/quadratic_data.csv"
data = np.genfromtxt(data_file, delimiter=',', names=True)
costs = data["cost"]
q1s = data["q1"]
q2s = data["q2"]
dq1s = data["dq1"]
dq2s = data["dq2"]
Deltas = data["Delta"]

num_iters = len(q1s);
gs = [ np.array([data["g1"][j], data["g2"][j]]) for j in range(num_iters)]
Hs = [ np.array([[data["H11"][j], data["H12"][j]], [data["H21"][j], data["H22"][j]]]) for j in range(num_iters)]

# Set up a plot
fig = plt.figure(figsize=(8,11))
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212)

# Make a 3D surface plot on the first subplot
ax1.plot_trisurf(q1, q2, cost, alpha=0.8)
ax1.set_xlabel("$q_1$")
ax1.set_ylabel("$q_2$")
ax1.set_zlabel("$L(q)$")

# Plot the path the optimizer took
ax1.plot(q1s, q2s, costs, "o-", color='red')

# Make a contour plot on the second subplot
levels = np.logspace(np.log10(min(cost)), np.log10(max(cost)), 30)
cp = ax2.contour(q1.reshape((50,50)),
            q2.reshape((50,50)),
            cost.reshape((50,50)),
            levels=levels,
            locator=ticker.LogLocator(30))
ax2.set_xlabel("$q_1$")
ax2.set_ylabel("$q_2$")
cb = fig.colorbar(cp, shrink=0.8, format="%1.1e")
cb.ax.set_title("L(q)")

# Search direction arrow dq
ax2.plot(q1s[i], q2s[i], "o-", color="red")
ax2.arrow(q1s[i], q2s[i], dq1s[i], dq2s[i], color="red")

# Gradient descent direction arrow
ax2.arrow(q1s[i], q2s[i], -1e5*gs[i][0], -1e5*gs[i][1], color="blue")

# Trust region circle
trust_region = plt.Circle((q1s[i],q2s[i]), Deltas[i], color='green', fill=False)
ax2.add_patch(trust_region)

# Quadratic approximation ellipse



plt.show()