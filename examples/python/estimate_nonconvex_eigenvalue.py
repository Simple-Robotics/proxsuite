import proxsuite
import numpy as np
import scipy.sparse as spa


def generate_mixed_qp(n, seed=1, reg=-2.0):
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
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
# generate a random non convex QP
H, g, A, b, C, u, l = generate_mixed_qp(n)
# initialize the model of the problem to solve
estimate_minimal_eigen_value = (
    proxsuite.proxqp.dense.estimate_minimal_eigen_value_of_symmetric_matrix(
        H, proxsuite.proxqp.EigenValueEstimateMethodOption.ExactMethod
    )
)
qp.init(H, g, A, b, C, l, u, manual_minimal_H_eigenvalue=estimate_minimal_eigen_value)
vals, _ = spa.linalg.eigs(H, which="SR")
min_eigenvalue = float(np.min(vals))
# print the estimates
print(f"{min_eigenvalue=}")
print(f"{estimate_minimal_eigen_value=}")
print(f"{qp.results.info.minimal_H_eigenvalue_estimate=}")
