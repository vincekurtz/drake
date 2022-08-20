#!/usr/bin/env python

##
#
# Make a plot of the cost landscape for a super simple system with two DoFs and
# one timestep (2 total decision variables).
#
##

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Get data for the contour plot
# N.B. Must run this script from the drake/ directory (containing bazel-out symlink)
drake_root = os.getcwd();
data_file = drake_root + "/bazel-out/k8-opt/bin/traj_opt/examples/2dof_spinner.runfiles/drake/contour_data.csv"

data = np.genfromtxt(data_file, delimiter=',', names=True)
q1 = data["q1"]
q2 = data["q2"]
cost = data["L"]

# Get iteration data (q_k, L_k, Delta_k, g_k, h_k)

# Make a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot_trisurf(q1,q2,cost)
ax.set_zlim((1e1,1e6))

# Make a contour plot
plt.subplot(2,1,2)
levels = np.logspace(1, 5.5, 30)
plt.tricontourf(q1,q2,cost, levels=levels)

print("Minimum cost: %s" % min(cost))

plt.show()