import proxsuite
import numpy as np
from util import generate_mixed_qp


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
# generate a random QP
H, g, A, b, C, u, l = generate_mixed_qp(n)
# initialize the model of the problem to solve
qp.init(H, g, A, b, C, l, u)
# solve without warm start
qp.solve()
# solve with a warm start, for ex random one
qp.solve(np.random.randn(n), np.random.randn(n_eq), np.random.randn(n_in))
# print an optimal solution
print("optimal x: {}".format(qp.results.x))
print("optimal y: {}".format(qp.results.y))
print("optimal z: {}".format(qp.results.z))
# Another example if you have box constraints (for the dense backend only for the moment)
qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
l_box = -np.ones(n) * 1.0e10
u_box = np.ones(n) * 1.0e10
qp2.init(H, g, A, b, C, l, u, l_box, u_box)
qp2.solve()
# An important note regarding the inequality multipliers
z_ineq = qp.results.z[:n_in]  # contains the multiplier associated to qp_random.C
z_box = qp.results.z[
    n_in:
]  # the last dim elements correspond to multiplier associated to
# the box constraints
