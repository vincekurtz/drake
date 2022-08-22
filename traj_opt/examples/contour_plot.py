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
g1 = data["g1"]
g2 = data["g2"]
H11 = data["H11"]
H22 = data["H22"]

# Plot cost, gradient, and Hessian for a particular slice 
ns = 150    # number of sample points in each axis
q1_square = q1.reshape((ns,ns))
q2_square = q2.reshape((ns,ns))
cost_square = cost.reshape((ns,ns))
g1_square = g1.reshape((ns,ns))
g2_square = g2.reshape((ns,ns))
H11_square = H11.reshape((ns,ns))
H22_square = H22.reshape((ns,ns))

plt.figure()
plt.subplot(311)
plt.title(f"Slice with q2 = {q2_square[0,0]}")
plt.plot(q1_square[:,0], cost_square[:,0])
plt.xlabel("q1")
plt.ylabel("cost")
plt.yscale("log")

plt.subplot(312)
plt.plot(q1_square[:,0], g1_square[:,0])
plt.xlabel("q1")
plt.ylabel("g1")

plt.subplot(313)
plt.plot(q1_square[:,0], H11_square[:,0])
plt.xlabel("q1")
plt.ylabel("H11")
plt.yscale("log")

plt.figure()
plt.subplot(311)
plt.title(f"Slice with q1 = {q1_square[0,0]}")
plt.plot(q2_square[0,:], cost_square[0,:])
plt.xlabel("q1")
plt.ylabel("cost")
plt.yscale("log")

plt.subplot(312)
plt.plot(q2_square[0,:], g2_square[0,:])
plt.xlabel("q2")
plt.ylabel("g2")

plt.subplot(313)
plt.plot(q2_square[0,:], H22_square[0,:])
plt.xlabel("q2")
plt.ylabel("H22")
plt.yscale("log")

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

## Make a 3D surface plot on the first subplot
#ax1.plot_trisurf(q1, q2, cost, alpha=0.8)
#ax1.set_xlabel("$q_1$")
#ax1.set_ylabel("$q_2$")
#ax1.set_zlabel("$L(q)$")
#ax1.set_xlim((min(q1), max(q1)))
#ax1.set_ylim((min(q2), max(q2)))
#
## Plot the path the optimizer took
#ax1.plot(q1s, q2s, costs, "o-", color='red')

# Make a contour plot on the second subplot
levels = np.logspace(np.log10(min(cost)), np.log10(max(cost)), 30)
cp = ax2.contour(q1.reshape((ns,ns)),
            q2.reshape((ns,ns)),
            cost.reshape((ns,ns)),
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
#ax2.arrow(q1s[i], q2s[i], -1e5*gs[i][0], -1e5*gs[i][1], color="blue")

# Trust region circle
trust_region = plt.Circle((q1s[i],q2s[i]), Deltas[i], color='green', fill=False)
ax2.add_patch(trust_region)

#print(f"Trust region radius: {Deltas[i]}")


plt.show()