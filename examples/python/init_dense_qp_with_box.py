import proxsuite
import numpy as np
from util import generate_mixed_qp


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
# you can check that there are constraints with method is_box_constrained
print(f"the qp is box constrained: {qp.is_box_constrained()}")
# generate a random QP
H, g, A, b, C, u, l = generate_mixed_qp(n)
l_box = -np.ones(n) * 1.0e10
u_box = np.ones(n) * 1.0e10
# initialize the model of the problem to solve
qp.init(H, g, A, b, C, l, u, l_box, u_box)
