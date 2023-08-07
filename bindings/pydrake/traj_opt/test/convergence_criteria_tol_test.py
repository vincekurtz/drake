# -*- coding: utf-8 -*-

import pydrake.traj_opt as mut

tol = mut.ConvergenceCriteriaTolerances()
print(tol.rel_cost_reduction)
print(tol.abs_cost_reduction)
print(tol.rel_gradient_along_dq)
print(tol.abs_gradient_along_dq)
print(tol.rel_state_change)
print(tol.abs_state_change)