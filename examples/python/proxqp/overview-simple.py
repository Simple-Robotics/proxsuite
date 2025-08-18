import proxsuite
from util import generate_mixed_qp


# generate a qp problem
n = 10
H, g, A, b, C, u, l = generate_mixed_qp(n)
n_eq = A.shape[0]
n_in = C.shape[0]

# solve it
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
qp.init(H, g, A, b, C, l, u)
qp.solve()
# print an optimal solution
print("optimal x: {}".format(qp.results.x))
print("optimal y: {}".format(qp.results.y))
print("optimal z: {}".format(qp.results.z))
