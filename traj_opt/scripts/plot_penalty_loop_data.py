#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font sizes for the plots
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=11)

##
#
# Script to plot the impact of the unactuation penalty on the
# constraint violation and the cost change. The data are obtained
# by enabling augmented_lagrangian and disabling update_init_guess
# as well as commenting out the lambda update in trajectory_optimizer.cc
# and setting the corresponding R values to zero.
#
##

# Select examples to be processed
examples = [
    '2dof_spinner',
    'acrobot',
    'allegro_hand',
    'allegro_hand_upside_down',
    'block_push',
    'frictionless_spinner',
    'hopper',
    'punyo_hug',
    'spinner'
]
num_examples = len(examples)

# Penalty values
max_major_iters = 10
penalty_vals = [1e1**i for i in range(max_major_iters)]

# Data to be parsed
iters = {}
costs = {}
violations = {}
minor_iters = {}
solve_times = {}
reasons = {}

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()
# get log file path
for ex in examples:
    data_file = drake_root + \
                "/bazel-out/k8-opt/bin/traj_opt/examples/analysis/" + \
                "penalty_loop/" + ex + ".runfiles/drake/al_solver_stats.csv"
    print(f"Reading the log file for {ex}: {data_file}")
    # Read data from the file and format nicely
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    print(data)
    iters[ex] = data["iter"]
    costs[ex] = data["cost"]
    violations[ex] = data["violation"]
    minor_iters[ex] = data["num_iters"]
    solve_times[ex] = data["time"]
    reasons[ex] = data["reason"]


fig, axs = plt.subplots(2, 1, figsize=(8, 8))
colors = plt.cm.jet(np.linspace(0, 1, num_examples))
for i, ex in enumerate(examples):
    # Convert the example data into numpy array
    ex_violation = np.array(violations[ex])
    ex_cost = np.array(costs[ex])
    # Normalize
    # ex_violation /= ex_violation[0]
    # ex_cost /= ex_cost[0]
    ex_violation /= np.max(ex_violation)
    ex_cost /= np.max(ex_cost)
    # Plot
    axs[0].plot(np.log10(penalty_vals), ex_violation, color=colors[i], marker='.')
    axs[1].plot(np.log10(penalty_vals), ex_cost, color=colors[i], marker='.')
# Add grid
for ax in axs:
    ax.grid()
# Set axis positions
axs[0].set_position([0.08, 0.52, 0.59, 0.45])
axs[1].set_position([0.08, 0.06, 0.59, 0.45])
# Add labels
axs[0].set_ylabel('Normalized unactuation violation')
axs[0].set_xticklabels([])
axs[1].set_ylabel('Normalized final position cost')
axs[1].set_xlabel('log10(Unactuation penalty)')
# Add legend
axs[0].legend(examples, bbox_to_anchor=(0.99, 0.5), loc='center left',
              frameon=False)
# Show the figure
plt.show()
