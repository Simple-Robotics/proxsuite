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
qp.init(H, g, A, b, C, l, u, rho=1.0e-7, mu_eq=1.0e-4)
qp.solve()
# If we redo a solve, qp.settings.default_rho value = 1.e-7, hence qp.results.info.rho restarts at 1.e-7
# The same occurs for mu_eq.
qp.solve()
# There might be a different result with WARM_START_WITH_PREVIOUS_RESULT initial guess option, as
# by construction, it reuses the last proximal step sizes of the last solving method.
