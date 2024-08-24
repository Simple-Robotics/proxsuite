import proxsuite
import numpy as np
import scipy.sparse as spa
from util import generate_mixed_qp


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
