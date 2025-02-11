import proxsuite
from util import generate_mixed_qp


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
qp = proxsuite.proxqp.sparse.QP(n, n_eq, n_in)
# generate a random QP
H, g, A, b, C, u, l = generate_mixed_qp(n, True)
# initialize the model of the problem to solve
qp.init(H, g, A, b, C, l, u)
qp.solve()
H_new = 2 * H  # keep the same sparsity structure
qp.update(H_new)  # update H with H_new, it will work
qp.solve()
# generate a QP with another sparsity structure
# create a new problem and update qp
H2, g_new, A_new, b_new, C_new, u_new, l_new = generate_mixed_qp(n, True)
qp.update(H=H2)  # nothing will happen
qp.update(g=g_new)  # if only a vector changes, then the update takes effect
qp.solve()  # it solves the problem with the QP H,g_new,A,b,C,u,l
# to solve the problem with H2 matrix create a new qp object in the sparse case
qp2 = proxsuite.proxqp.sparse.QP(n, n_eq, n_in)
qp2.init(H2, g_new, A, b, C, l, u)
qp2.solve()  # it will solve the new problem
# print an optimal solution
print("optimal x: {}".format(qp.results.x))
print("optimal y: {}".format(qp.results.y))
print("optimal z: {}".format(qp.results.z))
