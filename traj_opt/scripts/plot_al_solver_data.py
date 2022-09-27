#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os

from problem_properties import R, nominal_cost

##
#
# Script to plot the data obtained from the augmented Lagrangian
# solver with respect to the baseline obtained by running each
# example using the configuration. The data is pulled from a local
# directory specified by LOG_DIR. The user can select examples from
# the list by commenting out undesired lines.
#
##

# Select the log directory located at:
# drake/bazel-out/k8-opt/bin/traj_opt/examples/analysis/
# LOG_DIR = "al_default/"
# LOG_DIR = "al_custom_tol/"
# LOG_DIR = "al_mu0_1e2/"
# LOG_DIR = "al_mu0_1e3/"
# LOG_DIR = "al_mu0_1e4/"
LOG_DIR = "al_mu0_1e5/"

# Corresponding initial penalty value for the AL solver
MU0 = 1e5

# x axis selection
# 0: log10 of the unactuation penalty
# 1: the number of minor iterations
X_SELECT = 1

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
penalty_vals = np.array([1e1**i for i in range(max_major_iters)])
if LOG_DIR.startswith("al_mu0"):
    penalty_vals *= MU0

# Data to be parsed
iters = {}
costs = {}
violations = {}
minor_iters = {}
solve_times = {}
reasons = {}
baseline_iter = {}
baseline_cost = {}
baseline_violation = {}
# Post-processed data
total_minor_iters = {}

# drake/ directory, contains drake/bazel-out symlink
drake_root = os.getcwd()
# get log file path
for ex in examples:
    data_file = drake_root + \
                "/bazel-out/k8-opt/bin/traj_opt/examples/analysis/" + \
                LOG_DIR + ex + ".runfiles/drake/al_solver_stats.csv"
    print(f"\nReading the log file for {ex}: {data_file}")
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    print(data)
    # Parse the data
    try:
        iters[ex] = list(map(int, data["iter"]))
    except TypeError:
        iters[ex] = [int(data["iter"])]
    costs[ex] = data["cost"]
    violations[ex] = data["violation"]
    try:
        minor_iters[ex] = list(map(int, data["num_iters"]))
    except TypeError:
        minor_iters[ex] = [int(data["num_iters"])]
    solve_times[ex] = data["time"]
    try:
        reasons[ex] = list(map(int, data["reason"]))
    except:
        reasons[ex] = int(data["reason"])
    # Read and parse the baseline data
    baseline_file = drake_root + \
                    "/bazel-out/k8-opt/bin/traj_opt/examples/analysis/" + \
                    "baseline/" + ex + ".runfiles/drake/al_solver_stats.csv"
    print("Baseline data:")
    baseline_data = np.genfromtxt(baseline_file, delimiter=',', names=True)
    print(baseline_data)
    baseline_iter[ex] = baseline_data["num_iters"]
    baseline_cost[ex] = baseline_data["cost"]
    baseline_violation[ex] = baseline_data["violation"]

    # Post process
    total_minor_iters[ex] = []
    for i in iters[ex]:
        total_minor_iter = minor_iters[ex][i]
        if i > 0:
            total_minor_iter += total_minor_iters[ex][i-1]
        total_minor_iters[ex].append(total_minor_iter)


# Plot the data
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
colors = plt.cm.jet(np.linspace(0, 1, num_examples))
for i, ex in enumerate(examples):
    # Convert the example data into numpy array
    ex_num_iters = len(iters[ex])
    ex_violation = np.array(violations[ex])
    ex_cost = np.array(costs[ex])
    bl_violation = baseline_violation[ex]
    bl_cost = baseline_cost[ex]

    # Select x data
    if X_SELECT==0:
        x_data = np.log10(penalty_vals[:ex_num_iters])
        x_baseline = np.log10(R[ex])
        axs[1].set_xlabel('log10(unactuation penalty)')
    else:
        x_data = total_minor_iters[ex]
        x_baseline = baseline_iter[ex]
        axs[1].set_xlabel('Minor iterations')

    # Normalize y data
    # normalize_cost_by = np.max(ex_cost)
    normalize_cost_by = nominal_cost[ex]
    # normalize_violation_by = np.max(ex_violation)
    
    # ex_violation /= normalize_violation_by
    ex_cost /= normalize_cost_by
    # bl_violation /= normalize_violation_by
    bl_cost /= normalize_cost_by
    
    # Plot the baseline
    axs[0].plot(x_baseline, bl_violation, color=colors[i], marker='x')
    axs[1].plot(x_baseline, bl_cost, color=colors[i], marker='x')

    # Plot the AL data
    axs[0].plot(x_data, ex_violation, color=colors[i], marker='.', label=ex)
    axs[1].plot(x_data, ex_cost, color=colors[i], marker='.')

# Set axis limits and add grid
for ax in axs:
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid()
axs[1].set_ylim(top=1)
# Set axis positions
axs[0].set_position([0.075, 0.56, 0.62, 0.4])
axs[1].set_position([0.075, 0.1, 0.62, 0.4])
# Set labels and titles
axs[0].set_ylabel('Unactuation violation')
axs[0].set_xticklabels([])
axs[1].set_ylabel('Normalized final position cost')
axs[0].set_title(f"Data: {LOG_DIR}")
# Add legend
lgnd = axs[0].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                     frameon=False)
# Show the figure
plt.show()
