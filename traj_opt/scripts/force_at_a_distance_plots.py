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

F = 1.0
delta = 0.01
k = F/delta
s = 100.0

phi = np.linspace(-0.1,0.5, 1000)
fn = k*np.maximum(0, -phi)
fn_smooth = k/s * np.log(1+np.exp(-phi*s))

plt.plot(phi, fn_smooth, label="with force at a distance")
plt.plot(phi, fn, label="original contact model")
plt.axhline(0.0, color='grey', linestyle='--')
plt.axvline(0.0, color='grey', linestyle='--')

plt.xlabel("Signed Distance (m)")
plt.ylabel("Normal Force (N)")

plt.legend()

plt.show()
