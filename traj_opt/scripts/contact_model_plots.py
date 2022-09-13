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

# normal force
plt.subplot(2,1,1)
F = 1.0
delta = 0.01

phi = np.linspace(-0.05,0.02, 1000)
fn_1 = F*np.maximum(0, -phi/delta)**1
fn_15 = F*np.maximum(0, -phi/delta)**1.5
fn_2 = F*np.maximum(0, -phi/delta)**2
fn_4 = F*np.maximum(0, -phi/delta)**4

plt.plot(phi, fn_1, label="n = 1")
plt.plot(phi, fn_15, label="n = 1.5")
plt.plot(phi, fn_2, label="n = 2")
plt.plot(phi, fn_4, label="n = 4")
plt.axvline(0.0, color='grey', linestyle='--')
plt.axhline(0.0, color='grey', linestyle='--')

plt.xlabel("Signed Distance $\phi$ (m)")
plt.ylabel("Normal Force $f_n$ (N)")

plt.ylim((-0.5,10))
plt.legend()

# tangential force
plt.subplot(2,1,2)

fn = 1.0
mu = 1.0

vt = np.linspace(-0.5,0.5,1000)

vs = 0.1
ft_1 = mu*fn*vt / (vs * np.sqrt(1 + (vt/vs)**2))
vs = 0.05
ft_2 = mu*fn*vt / (vs * np.sqrt(1 + (vt/vs)**2))
vs = 0.01
ft_3 = mu*fn*vt / (vs * np.sqrt(1 + (vt/vs)**2))
vs = 0.005
ft_4 = mu*fn*vt / (vs * np.sqrt(1 + (vt/vs)**2))

plt.plot(vt, ft_1, label="$v_s$ = 0.1")
plt.plot(vt, ft_2, label="$v_s$ = 0.05")
plt.plot(vt, ft_3, label="$v_s$ = 0.01")
plt.plot(vt, ft_4, label="$v_s$ = 0.005")

plt.xlabel("Tangential Velocity $v_t$ (m/s)")
plt.ylabel("Tangential Force $f_t$ (N)")

plt.axhline(-1.0, color='grey', linestyle='--')
plt.axhline(1.0, color='grey', linestyle='--')

plt.yticks([-1.0,1.0],["$-\mu f_n$", "$\mu f_n$"])

plt.legend()


plt.show()
