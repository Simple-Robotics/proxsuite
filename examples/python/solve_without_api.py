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
H, g, A, b, C, u, l = generate_mixed_qp(n)

# solve the problem using the sparse backend
results = proxsuite.proxqp.sparse.solve(H, g, A, b, C, l, u)

# solve the problem using the dense backend

results2 = proxsuite.proxqp.dense.solve(
    H.toarray(), g, A.toarray(), b, C.toarray(), l, u
)
# Note finally, that the matrices are in sparse format, when using the dense backend you
# should convert them in dense format

# print an optimal solution
print("optimal x: {}".format(results.x))
print("optimal y: {}".format(results.y))
print("optimal z: {}".format(results.z))
