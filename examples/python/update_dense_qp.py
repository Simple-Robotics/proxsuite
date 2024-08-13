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
# create a new problem and update qp
H_new, g_new, A_new, b_new, C_new, u_new, l_new = generate_mixed_qp(n, seed=2)
qp.update(H_new, g_new, A_new, b_new, C_new, l_new, u_new)
# solve it
qp.solve()
# print an optimal solution
print("optimal x: {}".format(qp.results.x))
print("optimal y: {}".format(qp.results.y))
print("optimal z: {}".format(qp.results.z))
# if you have boxes (dense backend only) you proceed the same way
qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
l_box = -np.ones(n) * 1.0e10
u_box = np.ones(n) * 1.0e10
qp2.init(H, g, A, b, C, l, u, l_box, u_box)
qp2.solve()
l_box += 1.0e1
u_box -= 1.0e1
qp2.update(H_new, g_new, A_new, b_new, C_new, l_new, u_new, l_box, u_box)
qp2.solve()
