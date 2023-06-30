#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

import numpy as np
import proxsuite

# Linear program: min g^T.x such that C x <= u
g = np.array([1.0, 2.0, 3.0])
C = np.array(
    [
        [1.0, 2.0, -1.0],
        [2.0, 0.0, 1.0],
        [1.0, 2.0, 1.0],
        [-1.0, -1.0, -1.0],
    ]
)
u = np.array([4.0, 1.0, 3.0, 2.0])

# Initialize ProxQP problem
problem = proxsuite.proxqp.dense.QP(
    n=g.shape[0],
    n_eq=0,
    n_in=u.shape[0],
    # box_constraints = False,
    hessian_type=proxsuite.proxqp.dense.HessianType.Zero,
)
problem.settings.eps_abs = 1.0e-9
problem.init(None, g, None, None, C, None, u)

# Solve problem and print solution
problem.solve()
print("optimal x: {}".format(problem.results.x))
print("optimal y: {}".format(problem.results.y))
print("optimal z: {}".format(problem.results.z))
