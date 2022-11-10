#!/bin/bash

##
#
# Quick script for generating a convergence plots to evaluate the effectiveness
# of quasi-newton methods.
#
##

if [ ${PWD: -6} != "/drake" ]
then
    echo "This script must be run from the 'drake' directory";
    exit 0;
fi

# Set stiction velocities and quasi-newton flags to use
vs_list=(0.0001 0.0005 0.001 0.005 0.01 0.05)
qn_list=("true" "false")

# Run the experiments
for vs in ${vs_list[@]}
do
    for quasi_newton in ${qn_list[@]}
    do
        # Set parameters in the yaml file
        sed -i "s/stiction_velocity : .*/stiction_velocity : $vs/" ./traj_opt/examples/airhockey.yaml
        sed -i "s/quasi_newton : .*/quasi_newton : $quasi_newton/" ./traj_opt/examples/airhockey.yaml

        # Run the optimizer
        bazel run //traj_opt/examples:airhockey

        # Make a copy of the log files
        log_dir="./bazel-out/k8-opt/bin/traj_opt/examples/airhockey.runfiles/drake/"
        original_log="solver_stats.csv"
        new_log="solver_stats_vs_${vs}_quasi_newton_${quasi_newton}.csv"
        cp ${log_dir}${original_log} ${log_dir}${new_log}
    done
done

# Run the python script to generate comparison plots from these logs
./traj_opt/scripts/quasi_newton_airhockey_comparison.py
