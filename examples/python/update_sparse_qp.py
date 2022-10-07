import proxsuite
import numpy as np
import scipy.sparse as spa


def generate_mixed_qp(n, seed=1):
    # A function for generating random convex qps

    np.random.seed(seed)
    n_eq = int(n / 4)
    n_in = int(n / 4)
    m = n_eq + n_in

    P = spa.random(
        n, n, density=0.075, data_rvs=np.random.randn, format="csc"
    ).toarray()
    P = (P + P.T) / 2.0

    s = max(np.absolute(np.linalg.eigvals(P)))
    P += (abs(s) + 1e-02) * spa.eye(n)
    P = spa.coo_matrix(P)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    v = np.random.randn(n)  # Fictitious solution
    delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P, q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
qp = proxsuite.proxqp.sparse.QP(n, n_eq, n_in)
# generate a random QP
H, g, A, b, C, u, l = generate_mixed_qp(n)
# initialize the model of the problem to solve
qp.init(H, g, A, b, C, l, u)
qp.solve()
H_new = 2 * H  # keep the same sparsity structure
qp.update(H_new)  # update H with H_new, it will work
qp.solve()
# generate a QP with another sparsity structure
# create a new problem and update qp
H2, g_new, A_new, b_new, C_new, u_new, l_new = generate_mixed_qp(n)
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
