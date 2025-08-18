import proxsuite
from util import generate_mixed_qp


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
# generate a random QP
H, g, A, b, C, u, l = generate_mixed_qp(n)
# initialize the model of the problem to solve
qp.settings.compute_timings  # compute all timings
qp.init(H, g, A, b, C, l, u)
