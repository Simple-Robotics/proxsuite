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
qp.init(H, g, A, b, C, l, u)
# solve without warm start
qp.solve()
# create a new problem and update qp
g_new = 0.95 * g  # slightly different g_new
qp.settings.initial_guess = (
    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
)
qp.update(g=g_new)
# solve it
qp.solve()
# print an optimal solution
print("optimal x: {}".format(qp.results.x))
print("optimal y: {}".format(qp.results.y))
print("optimal z: {}".format(qp.results.z))
