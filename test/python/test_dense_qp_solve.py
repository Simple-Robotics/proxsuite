#
# Copyright (c) 2022, INRIA
#
import os
import proxsuite
import numpy as np
import scipy.sparse as spa
import scipy.io as spio
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

    P = spa.random(
        n, n, density=0.075, data_rvs=np.random.randn, format="csc"
    ).toarray()
    P = (P + P.T) / 2.0

    s = max(np.absolute(np.linalg.eigvals(P)))
    P += (abs(s) + 1e-02) * spa.eye(n)
    P = spa.coo_matrix(P)
    # print("sparsity of P : {}".format((P.nnz) / (n**2)))
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc").toarray(
        order="C"
    )  # row-major
    v = np.random.randn(n)  # Fictitious solution
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


class DenseQpWrapper(unittest.TestCase):
    # TESTS DENSE SOLVE FUNCTION

    def test_case_basic_solve(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test basic solve"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        results = proxsuite.proxqp.dense.solve(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            eps_abs=1.0e-9,
            eps_rel=0,
        )
        dua_res = normInf(
            H @ results.x + g + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_case_different_rho_value(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test different rho values"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        results = proxsuite.proxqp.dense.solve(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            eps_abs=1.0e-9,
            eps_rel=0,
            rho=1.0e-7,
        )
        dua_res = normInf(
            H @ results.x + g + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert results.info.rho == 1e-7
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_case_different_mu_values(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test different mu_eq and mu_in values"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        results = proxsuite.proxqp.dense.solve(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            eps_abs=1.0e-9,
            eps_rel=0,
            mu_eq=1.0e-2,
            mu_in=1.0e-2,
        )
        dua_res = normInf(
            H @ results.x + g + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_case_different_warm_starting(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test warm starting"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        x_wm = np.random.randn(n)
        y_wm = np.random.randn(n_eq)
        z_wm = np.random.randn(n_in)
        results = proxsuite.proxqp.dense.solve(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            eps_abs=1.0e-9,
            eps_rel=0,
            x=x_wm,
            y=y_wm,
            z=z_wm,
        )
        dua_res = normInf(
            H @ results.x + g + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_case_different_verbose_true(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test verbose = true"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        results = proxsuite.proxqp.dense.solve(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            eps_abs=1.0e-9,
            eps_rel=0,
            verbose=True,
        )
        dua_res = normInf(
            H @ results.x + g + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_case_different_no_initial_guess(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test no initial guess"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        results = proxsuite.proxqp.dense.solve(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            eps_abs=1.0e-9,
            eps_rel=0,
            initial_guess=proxsuite.proxqp.NO_INITIAL_GUESS,
        )
        dua_res = normInf(
            H @ results.x + g + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_sparse_problem_with_exact_solution_known(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints and exact solution known"
        )

        n = 150
        M = spa.lil_matrix(spa.eye(n))
        for i in range(1, n - 1):
            M[i, i + 1] = -1
            M[i, i - 1] = 1

        H = spa.csc_matrix(M.dot(M.transpose())).toarray()
        H = np.ascontiguousarray(H)
        g = -np.ones((n,))
        A = None
        b = None
        C = spa.csc_matrix(spa.eye(n)).toarray()
        C = np.ascontiguousarray(C)
        l = 2.0 * np.ones((n,))
        u = np.full(l.shape, +np.inf)

        results = proxsuite.proxqp.dense.solve(H, g, A, b, C, l, u)
        x_theoretically_optimal = np.array([2.0] * 149 + [3.0])

        dua_res = normInf(H @ results.x + g + C.transpose() @ results.z)
        pri_res = normInf(
            np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
        )

        assert dua_res <= 1e-3  # default precision of the solver
        assert pri_res <= 1e-3
        assert normInf(x_theoretically_optimal - results.x) <= 1e-3
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, 0, n))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_initializing_with_None(self):
        print("------------------------test initialization with Nones")

        H = np.array([[65.0, -22.0, -16.0], [-22.0, 14.0, 7.0], [-16.0, 7.0, 5.0]])
        g = np.array([-13.0, 15.0, 7.0])
        A = None
        b = None
        C = None
        _u = None
        _l = None

        results = proxsuite.proxqp.dense.solve(
            H,
            g,
            A,
            b,
            C,
        )
        print("optimal x: {}".format(results.x))

        dua_res = normInf(H @ results.x + g)

        assert dua_res <= 1e-3  # default precision of the solver
        print("--n = {} ; n_eq = {} ; n_in = {}".format(3, 0, 0))
        print("dual residual = {} ".format(dua_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )

    def test_solve_qpsolvers_problem(self):
        print(
            "------------------------test case from qpsolvers with equality constraint and upper bound inequality constraints"
        )
        file_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(file_path, "..", "data")
        m = spio.loadmat(
            os.path.join(data_path, "simple_qp_with_inifinity_lower_bound.mat"),
            squeeze_me=True,
        )
        P = np.ascontiguousarray(m["P"].astype(float))
        q = m["q"].astype(float)
        A = np.ascontiguousarray(m["A"].astype(float).reshape((1, 3)))
        b = np.array([m["b"]]).reshape((1,))
        C = np.ascontiguousarray(m["C"].astype(float))
        l = m["l"].astype(float)
        u = m["u"].astype(float)

        results = proxsuite.proxqp.dense.solve(P, q, A, b, C, l, u, verbose=False)
        print("optimal x: {}".format(results.x))

        dua_res = normInf(
            P @ results.x + q + A.transpose() @ results.y + C.transpose() @ results.z
        )
        pri_res = max(
            normInf(A @ results.x - b),
            normInf(
                np.maximum(C @ results.x - u, 0) + np.minimum(C @ results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-5
        assert pri_res <= 1e-5

        print("--n = {} ; n_eq = {} ; n_in = {}".format(3, 1, 3))
        print("dual residual = {} ".format(dua_res))
        print("total number of iteration: {}".format(results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                results.info.setup_time, results.info.solve_time
            )
        )


if __name__ == "__main__":
    unittest.main()
