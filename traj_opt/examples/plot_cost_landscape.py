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

# Set up a plot
fig = plt.figure(figsize=(8,11))
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212)

# Make a 3D surface plot on the first subplot
ax1.plot_trisurf(q1,q2,cost)
#ax1.set_zlim((1e1,1e6))

# Make a contour plot on the second subplot
levels = np.logspace(1, 7, 30)
plt.contour(q1.reshape((50,50)),
            q2.reshape((50,50)),
            cost.reshape((50,50)),
            levels=levels)

print("Minimum cost: %s" % min(cost))

plt.show()