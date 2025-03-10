#
# Copyright (c) 2023, INRIA
#
import proxsuite
import numpy as np
import scipy.sparse as spa
import unittest


def normInf(x):
    if x.shape[0] == 0:
        return 0.0
    else:
        return np.linalg.norm(x, np.inf)


def generate_mixed_qp(n, seed=1):
    """
    Generate sparse problem in dense QP format
    """
    np.random.seed(seed)

    m = int(n / 4) + int(n / 4)
    # m  = n
    n_eq = int(n / 4)
    n_in = int(n / 4)

    P = spa.random(n, n, density=0.75, data_rvs=np.random.randn, format="csc").toarray()
    P = (P + P.T) / 2.0

    s = max(np.absolute(np.linalg.eigvals(P)))
    P += (abs(s) + 1e-02) * spa.eye(n)
    P = spa.coo_matrix(P)
    # print("sparsity of P : {}".format((P.nnz) / (n**2)))
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.95, data_rvs=np.random.randn, format="csc").toarray(
        order="C"
    )
    v = np.random.randn(n)  # Fictitious solution
    _delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return (
        P.toarray(order="C"),
        q,
        A[:n_eq, :],
        u[:n_eq],
        A[n_in:, :],
        u[n_in:],
        l[n_in:],
    )


class ParallelWrapper(unittest.TestCase):
    # TESTS OF GENERAL METHODS OF THE API

    def test_dense_parallel(self):
        n = 10  # dimension
        batch_size = 4
        qps = []
        qps_compare = proxsuite.proxqp.dense.VectorQP()

        for i in range(batch_size):
            H, g, A, b, C, u, l = generate_mixed_qp(n, seed=i)
            n_eq = A.shape[0]
            n_in = C.shape[0]

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.settings.eps_abs = 1.0e-9
            qp.settings.verbose = False
            qp.init(
                H=H,
                g=np.asfortranarray(g),
                A=A,
                b=np.asfortranarray(b),
                C=C,
                l=np.asfortranarray(l),
                u=np.asfortranarray(u),
                rho=1.0e-7,
            )
            qps.append(qp)

            qp_compare = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp_compare.settings.eps_abs = 1.0e-9
            qp_compare.settings.verbose = False
            qp_compare.init(
                H=H,
                g=np.asfortranarray(g),
                A=A,
                b=np.asfortranarray(b),
                C=C,
                l=np.asfortranarray(l),
                u=np.asfortranarray(u),
                rho=1.0e-7,
            )
            qps_compare.append(qp_compare)

        for qp in qps:
            qp.solve()

        num_threads = proxsuite.proxqp.omp_get_max_threads() - 1
        proxsuite.proxqp.dense.solve_in_parallel(qps_compare, num_threads)

        for i in range(batch_size):
            assert np.allclose(qps[i].results.x, qps_compare[i].results.x, rtol=1e-8)

    def test_dense_parallel_custom_BatchQP(self):
        n = 10  # dimension
        batch_size = 4
        qps = []
        qp_vector = proxsuite.proxqp.dense.BatchQP()

        for i in range(batch_size):
            H, g, A, b, C, u, l = generate_mixed_qp(n, seed=i)
            n_eq = A.shape[0]
            n_in = C.shape[0]

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.settings.eps_abs = 1.0e-9
            qp.settings.verbose = False
            qp.init(
                H=H,
                g=np.asfortranarray(g),
                A=A,
                b=np.asfortranarray(b),
                C=C,
                l=np.asfortranarray(l),
                u=np.asfortranarray(u),
                rho=1.0e-7,
            )
            qps.append(qp)

            qp_compare = qp_vector.init_qp_in_place(n, n_eq, n_in)
            qp_compare.settings.eps_abs = 1.0e-9
            qp_compare.settings.verbose = False
            qp_compare.init(
                H=H,
                g=np.asfortranarray(g),
                A=A,
                b=np.asfortranarray(b),
                C=C,
                l=np.asfortranarray(l),
                u=np.asfortranarray(u),
                rho=1.0e-7,
            )

        for qp in qps:
            qp.solve()

        num_threads = proxsuite.proxqp.omp_get_max_threads() - 1
        proxsuite.proxqp.dense.solve_in_parallel(qp_vector, num_threads)

        for i in range(batch_size):
            assert np.allclose(qps[i].results.x, qp_vector.get(i).results.x, rtol=1e-8)

    def test_sparse_parallel_custom_BatchQP(self):
        n = 10  # dimension
        batch_size = 4
        qps = []
        qp_vector = proxsuite.proxqp.sparse.BatchQP()

        for i in range(batch_size):
            H, g, A, b, C, u, l = generate_mixed_qp(n, seed=i)
            _n_eq = A.shape[0]
            _n_in = C.shape[0]

            H_ = H != 0.0
            A_ = A != 0.0
            C_ = C != 0.0
            qp = proxsuite.proxqp.sparse.QP(H_, A_, C_)
            qp.settings.eps_abs = 1.0e-9
            qp.settings.verbose = False
            qp.init(
                H=H,
                g=np.asfortranarray(g),
                A=A,
                b=np.asfortranarray(b),
                C=C,
                l=np.asfortranarray(l),
                u=np.asfortranarray(u),
                rho=1.0e-7,
            )
            qps.append(qp)

            # qp_compare = qp_vector.init_qp_in_place(H_, A_, C_)
            qp_compare = qp_vector.init_qp_in_place(H.shape[0], A.shape[0], C.shape[0])
            qp_compare.settings.eps_abs = 1.0e-9
            qp_compare.settings.verbose = False
            qp_compare.init(
                H=H,
                g=np.asfortranarray(g),
                A=A,
                b=np.asfortranarray(b),
                C=C,
                l=np.asfortranarray(l),
                u=np.asfortranarray(u),
                rho=1.0e-7,
            )

        for qp in qps:
            qp.solve()

        num_threads = proxsuite.proxqp.omp_get_max_threads() - 1
        proxsuite.proxqp.sparse.solve_in_parallel(qp_vector, num_threads)

        for i in range(batch_size):
            assert np.allclose(qps[i].results.x, qp_vector.get(i).results.x, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
