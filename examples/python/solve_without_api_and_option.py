import proxsuite
import numpy as np
import scipy.sparse as spa


def generate_mixed_qp(n, seed=1):
    # A function for generating sparse random convex qps in dense format

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
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc").toarray()
    v = np.random.randn(n)  # Fictitious solution
    delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
H, g, A, b, C, u, l = generate_mixed_qp(n)
# solve the problem using the sparse backend
# and suppose you want to change the accuracy to 1.E-9 and rho initial value to 1.E-7
results = proxsuite.proxqp.dense.solve(
    H=H, g=g, A=A, b=b, C=C, l=l, u=u, rho=1.0e-7, eps_abs=1.0e-9
)
# Note that in python the order does not matter for rho and eps_abs
# print an optimal solution
print("optimal x: {}".format(results.x))
print("optimal y: {}".format(results.y))
print("optimal z: {}".format(results.z))
