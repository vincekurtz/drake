import numpy as np

##
#
# This script includes the hard-coded penalty and nominal cost values
# obtained from the default configuration file per example. The
# nominal cost is defined as the final position cost between the
# initial and goal poses.
#
##

# Nominal penalty values per example
R = {}
R['2dof_spinner'] = 1e6
R['acrobot'] = 1e3
R['allegro_hand'] = 10**(4.5)
R['allegro_hand_upside_down'] = 10**(4.5)
R['block_push'] = 1e3
R['frictionless_spinner'] = 1e6
R['hopper'] = 10**(3.5)
R['punyo_hug'] = 10**(3.5)
R['spinner'] = 1e6

# Final position cost from the initial pose
q_init = {}
q_goal = {}
q_diff = {}
Qf = {}
nominal_cost = {}
ex = '2dof_spinner'
q_init[ex] = np.array([1.3, 0.0])
q_goal[ex] = np.array([0.0, -1.5])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([10.0, 10.0])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'acrobot'
q_init[ex] = np.array([0.0, 0.0])
q_goal[ex] = np.array([3.1415, 0.0])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([100.0, 100.0])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'allegro_hand'
q_init[ex] = np.array([-0.1, 1.0, 1.0, 1.0, # outside finger
          0.6, 1.9, 1.0, 1.0,               # thumb
          0.0, 0.7, 1.0, 1.0,               # middle finger
          0.1, 1.0, 1.0, 1.0,               # inside finger
          1.0, 0.0, 0.0, 0.0,               # ball orientation
         -0.06, 0.0, 0.1])                  # ball position
q_goal[ex] = np.array([-0.1, 1.0, 1.0, 1.0,
              0.6, 1.9, 1.0, 1.0,
              0.0, 0.7, 1.0, 1.0,
              0.1, 1.0, 1.0, 1.0,
              0.7, 0.0, 0.0, 0.7,
             -0.06, 0.0, 0.1])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([0e0, 0e0, 0e0, 0e0,
       0e0, 0e0, 0e0, 0e0,
       0e0, 0e0, 0e0, 0e0,
       0e0, 0e0, 0e0, 0e0,
       1e2, 1e2, 1e2, 1e2,
       1e2, 1e2, 1e2])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'allegro_hand_upside_down'
q_init[ex] = np.array([-0.2, 1.4, 0.6, 0.7, # outside finger
           0.3, 1.5, 1.0, 1.0,              # thumb
           0.0, 0.7, 1.0, 1.0,              # middle finger
           0.1, 1.0, 1.0, 1.0,              # inside finger
           1.0, 0.0, 0.0, 0.0,              # ball orientation
          -0.06, 0.0, 0.07])                # ball position
q_goal[ex] = np.array([-0.2, 1.4, 0.6, 0.7,
              0.3, 1.5, 1.0, 1.0,
              0.0, 0.7, 1.0, 1.0,
              0.1, 1.0, 1.0, 1.0,
              0.7, 0.0, 0.0,-0.7,
             -0.06, 0.0, 0.07])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([0e0, 0e0, 0e0, 0e0,
       0e0, 0e0, 0e0, 0e0,
       0e0, 0e0, 0e0, 0e0,
       0e0, 0e0, 0e0, 0e0,
       1e2, 1e2, 1e2, 1e2,
       1e2, 1e2, 1e2])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'block_push'
q_init[ex] = np.array([-0.15, 0.0, 0.1,  1.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.096])
q_goal[ex] = np.array([-0.2, 0.0, 0.1,   0.9, 0.0, 0.0, 0.3,   0.5, 0.5, 0.1])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([0, 0, 0,   10, 10, 10, 10,   50, 50, 50])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'frictionless_spinner'
q_init[ex] = np.array([-0.1, 1.5, 0.0])
q_goal[ex] = np.array([0.0, 0.0,-3.1])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([0.0, 0.0, 10.0])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'hopper'
q_init[ex] = np.array([0.0, 0.61, 0.3,-0.5, 0.2])
q_goal[ex] = np.array([0.5, 0.61, 0.3,-0.5, 0.2])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([1e4, 1, 1, 10, 10])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'punyo_hug'
q_init[ex] = np.array([-0.1,        # height
          0.0, 0.0, 0.0,            # torso
          1.0, 0.1, 0.5, 0.0, 0.0,  # right arm
          1.0, 0.1, 0.5, 0.0, 0.0,  # left arm
          1.0, 0.0, 0.0, 0.0,       # ball orientation
          0.0, 0.3, 0.3])           # ball position
q_goal[ex] = np.array([0.0,
             0.0, 0.0, 0.0,
             1.0, 0.1, 0.5, 0.0, 0.0,
             1.0, 0.1, 0.5, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0,
             0.2, 0.3, 1.0])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([10, 
       0, 0, 0, 
       0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 
       1, 1, 1, 1,
       100, 100, 100])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])
ex = 'spinner'
q_init[ex] = np.array([-0.1, 1.5, 0.0])
q_goal[ex] = np.array([0.0, 0.0,-3.0])
q_diff[ex] = q_goal[ex] - q_init[ex]
Qf[ex] = np.diag([0.0, 0.0, 10.0])
nominal_cost[ex] = q_diff[ex].T @ (Qf[ex] @ q_diff[ex])