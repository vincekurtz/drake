#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

##
#
# Quick script to make a plot of the sparsity pattern of the Hessian.
#
# This script must be run from the "drake/" directory. 
#
##

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()

data_file = drake_root + "/bazel-out/k8-opt/bin/traj_opt/examples/spinner.runfiles/drake/hessian.csv"
data = np.genfromtxt(data_file, dtype=float, names=None)
H = np.asarray(data)

plt.imshow(np.log(np.abs(H)), cmap="binary")
plt.xticks([])
plt.yticks([])
plt.show()
