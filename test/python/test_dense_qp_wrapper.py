#
# Copyright (c) 2022-2023, INRIA
#
import proxsuite
import numpy as np
import scipy.sparse as spa
import unittest

np.printoptions(precision=16)


def normInf(x):
    if x.shape[0] == 0:
        return 0.0
    else:
        return np.linalg.norm(x, np.inf)


def generate_mixed_qp(n, seed=1, reg=0.01):
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
    P += (abs(s) + reg) * spa.eye(n)
    P = spa.coo_matrix(P)
    # print("sparsity of P : {}".format((P.nnz) / (n**2)))
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc").toarray(
        order="C"
    )
    v = np.random.randn(n)  # Fictitious solution
    _delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


def generate_mixed_qp_with_box(n, seed=1):
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
    )
    v = np.random.randn(n)  # Fictitious solution
    u = A @ v
    l = -1.0e20 * np.ones(m)
    delta_box = np.random.rand(n)
    u_box = v + delta_box
    l_box = v - delta_box

    return (
        P.toarray(),
        q,
        A[:n_eq, :],
        u[:n_eq],
        A[n_in:, :],
        u[n_in:],
        l[n_in:],
        u_box,
        l_box,
    )


class DenseqpWrapper(unittest.TestCase):
    # TESTS OF GENERAL METHODS OF THE API
    def test_case_deterministic_behavior(self):
        print("------------------------test the result is deterministic")
        n = 100
        H, g, A, b, C, u, l = generate_mixed_qp(n)
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
        )
        qp.solve()
        x_prev = np.copy(qp.results.x)
        y_prev = np.copy(qp.results.y)
        z_prev = np.copy(qp.results.z)
        for i in range(20):
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
            )
            qp.solve()

            print(f"{normInf(x_prev - qp.results.x)=}")
            print(f"{normInf(y_prev - qp.results.y)=}")
            print(f"{normInf(z_prev - qp.results.z)=}")

            assert normInf(x_prev - qp.results.x) <= 1e-14
            assert normInf(y_prev - qp.results.y) <= 1e-14
            assert normInf(z_prev - qp.results.z) <= 1e-14

    def test_case_update_rho(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test update rho"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
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
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_update_mu(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test update mus"
        )

        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            l=np.asfortranarray(l),
            u=np.asfortranarray(u),
            mu_eq=1.0e-2,
            mu_in=1.0e-3,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_no_equilibration_at_initialization(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test with no equilibration at initialization"
        )

        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            False,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_with_equilibration_at_initialization(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test with equilibration at initialization"
        )

        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_no_initial_guess(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test with no initial guess"
        )

        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_no_initial_guess_and_update(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test with no initial guess"
        )

        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        g = np.random.randn(n)
        H *= 2.0  # too keep same sparsity structure
        qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )

        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_starting(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test with warm start---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )
        x_wm = np.random.randn(n)
        y_wm = np.random.randn(n_eq)
        z_wm = np.random.randn(n_in)
        qp.solve(x_wm, y_wm, z_wm)

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_previous_result(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test with warm start with previous result---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1e-9
        assert dua_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )

        x = qp.results.x
        y = qp.results.y
        z = qp.results.z
        qp2.solve(x, y, z)

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp2".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_cold_start_with_previous_result(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test with cold start with previous result---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
        )
        x = qp.results.x
        y = qp.results.y
        z = qp.results.z
        qp2.solve(x, y, z)

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp2".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_equilibration_option(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test equilibration option---"
        )

        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            False,
        )
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with qp2".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_equilibration_option_at_update(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test equilibration option at update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print("--n = {} ; n_eq = {} ; n_in = {} with qp".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.update(update_preconditioner=True)
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} with qp after update".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            False,
        )
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print("--n = {} ; n_eq = {} ; n_in = {} with qp2".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2.update(update_preconditioner=False)
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} with qp2 after update".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_other_initialization(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test warm start with other initialization---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve(np.random.randn(n), np.random.randn(n_eq), np.random.randn(n_in))

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print("--n = {} ; n_eq = {} ; n_in = {} with qp".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    # TESTS ALL INITIAL GUESS OPTIONS FOR MULTIPLE SOLVES AT ONCE

    def test_case_multiple_solve_with_no_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve with no inital guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_multiple_solve_with_equality_constrained_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve with equality constrained initial guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_previous_result_starting_with_equality_constraints_initial_guess(
        self,
    ):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve after warm starting with previous results and equality constrained inital guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_previous_result_starting_with_no_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve after warm starting with previous results and no initial guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_cold_start_with_previous_result_starting_with_no_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve after cold starting with previous results and no initial guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_no_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve from warm start and no initial guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        qp.init(H, g, A, b, C, l, u)
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START

        qp.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_no_initial_guess_and_different_init(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test solve from warm start and no initial guess with other initialization---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        qp.init(H, g, A, b, C, l, u)
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.init(H, g, A, b, C, l, u)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        qp2.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve with new QP object")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp2.results.info.setup_time, qp2.results.info.solve_time
            )
        )

    # TESTS WITH UPDATE + INITIAL GUESS OPTIONS

    def test_case_multiple_solve_with_no_initial_guess_and_update(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve with no inital guess and update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            update_preconditioner,
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_multiple_solve_with_equality_constrained_initial_guess_and_update(
        self,
    ):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve with equality constrained initial guess and update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            update_preconditioner,
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_previous_result_starting_with_equality_constraints_initial_guess_and_update(
        self,
    ):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve after warm starting with previous results and equality constrained inital guess and update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            update_preconditioner,
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_previous_result_starting_with_no_initial_guess_and_update(
        self,
    ):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve after warm starting with previous results and no initial guess and update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            update_preconditioner,
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_cold_start_with_previous_result_starting_with_no_initial_guess_and_update(
        self,
    ):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve after cold starting with previous results and no initial guess and update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )

        H *= 2  # keep same sparsity structure
        g = np.random.randn(n)
        qp.update(H=H, g=np.asfortranarray(g), update_preconditioner=True)
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_warm_start_with_no_initial_guess_and_update(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test multiple solve from warm start and no initial guess and update---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            update_preconditioner,
        )
        qp.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp.solve(qp.results.x, qp.results.y, qp.results.z)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_case_initialization_with_rho_for_different_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test initializaton with rho for different initial guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=1.0e-7,
        )
        qp.solve()
        assert qp.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=1.0e-7,
        )
        qp2.solve()
        assert qp2.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp2.results.info.setup_time, qp2.results.info.solve_time
            )
        )

        qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp3.settings.eps_abs = 1.0e-9
        qp3.settings.verbose = False
        qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp3.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=1.0e-7,
        )
        qp3.solve()
        assert qp.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ qp3.results.x
            + g
            + A.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A @ qp3.results.x - b),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp3.results.info.setup_time, qp3.results.info.solve_time
            )
        )

        qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp4.settings.eps_abs = 1.0e-9
        qp4.settings.verbose = False
        qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp4.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=1.0e-7,
        )
        qp4.solve()
        assert qp4.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ qp4.results.x
            + g
            + A.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A @ qp4.results.x - b),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp4.results.info.setup_time, qp4.results.info.solve_time
            )
        )

        qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp5.settings.eps_abs = 1.0e-9
        qp5.settings.verbose = False
        qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp5.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=1.0e-7,
        )
        qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z)
        assert qp.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ qp5.results.x
            + g
            + A.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A @ qp5.results.x - b),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp5.results.info.setup_time, qp5.results.info.solve_time
            )
        )

    def test_case_update_g_for_different_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test update g for different initial guess---"
        )
        n = 10
        H, g_old, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()
        g = np.random.randn(n)
        dua_res = normInf(
            H @ qp.results.x
            + g_old
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(g=g)
        assert normInf(qp.model.g - g) <= 1.0e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp2.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g_old
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp2.update(g=g)
        assert normInf(qp.model.g - g) <= 1.0e-9
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp2.results.info.setup_time, qp2.results.info.solve_time
            )
        )

        qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp3.settings.eps_abs = 1.0e-9
        qp3.settings.verbose = False
        qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp3.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp3.solve()
        dua_res = normInf(
            H @ qp3.results.x
            + g_old
            + A.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A @ qp3.results.x - b),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp3.update(g=g)
        assert normInf(qp.model.g - g) <= 1.0e-9
        qp3.solve()
        dua_res = normInf(
            H @ qp3.results.x
            + g
            + A.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A @ qp3.results.x - b),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp3.results.info.setup_time, qp3.results.info.solve_time
            )
        )

        qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp4.settings.eps_abs = 1.0e-9
        qp4.settings.verbose = False
        qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp4.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp4.solve()
        dua_res = normInf(
            H @ qp4.results.x
            + g_old
            + A.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A @ qp4.results.x - b),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp4.update(g=g)
        assert normInf(qp.model.g - g) <= 1.0e-9
        qp4.solve()
        dua_res = normInf(
            H @ qp4.results.x
            + g
            + A.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A @ qp4.results.x - b),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp4.results.info.setup_time, qp4.results.info.solve_time
            )
        )

        qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp5.settings.eps_abs = 1.0e-9
        qp5.settings.verbose = False
        qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp5.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z)
        dua_res = normInf(
            H @ qp5.results.x
            + g_old
            + A.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A @ qp5.results.x - b),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp5.update(g=g)
        assert normInf(qp.model.g - g) <= 1.0e-9
        qp5.solve()
        dua_res = normInf(
            H @ qp5.results.x
            + g
            + A.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A @ qp5.results.x - b),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp5.results.info.setup_time, qp5.results.info.solve_time
            )
        )

    def test_case_update_A_for_different_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test update A for different initial guess---"
        )
        n = 10
        H, g, A_old, b_old, C, u, l = generate_mixed_qp(n)
        n_eq = A_old.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b_old),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()
        _, _, A_new, b_new, _, _, _ = generate_mixed_qp(n, seed=2)
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A_old.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A_old @ qp.results.x - b_old),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(A=A_new, b=b_new)
        assert normInf(qp.model.A - A_new) <= 1.0e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A_new.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A_new @ qp.results.x - b_new),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp2.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b_old),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A_old.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A_old @ qp2.results.x - b_old),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp2.update(A=A_new, b=b_new)
        assert normInf(qp.model.A - A_new) <= 1.0e-9
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A_new.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A_new @ qp2.results.x - b_new),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp2.results.info.setup_time, qp2.results.info.solve_time
            )
        )

        qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp3.settings.eps_abs = 1.0e-9
        qp3.settings.verbose = False
        qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp3.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b_old),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp3.solve()
        dua_res = normInf(
            H @ qp3.results.x
            + g
            + A_old.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A_old @ qp3.results.x - b_old),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp3.update(A=A_new, b=b_new)
        assert normInf(qp.model.A - A_new) <= 1.0e-9
        qp3.solve()
        dua_res = normInf(
            H @ qp3.results.x
            + g
            + A_new.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A_new @ qp3.results.x - b_new),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp3.results.info.setup_time, qp3.results.info.solve_time
            )
        )

        qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp4.settings.eps_abs = 1.0e-9
        qp4.settings.verbose = False
        qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp4.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b_old),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp4.solve()
        dua_res = normInf(
            H @ qp4.results.x
            + g
            + A_old.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A_old @ qp4.results.x - b_old),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp4.update(A=A_new, b=b_new)
        assert normInf(qp.model.A - A_new) <= 1.0e-9
        qp4.solve()
        dua_res = normInf(
            H @ qp4.results.x
            + g
            + A_new.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A_new @ qp4.results.x - b_new),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp4.results.info.setup_time, qp4.results.info.solve_time
            )
        )

        qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp5.settings.eps_abs = 1.0e-9
        qp5.settings.verbose = False
        qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp5.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b_old),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z)
        dua_res = normInf(
            H @ qp5.results.x
            + g
            + A_old.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A_old @ qp5.results.x - b_old),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp5.update(A=A_new, b=b_new)
        assert normInf(qp.model.A - A_new) <= 1.0e-9
        qp5.solve()
        dua_res = normInf(
            H @ qp5.results.x
            + g
            + A_new.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A_new @ qp5.results.x - b_new),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp5.results.info.setup_time, qp5.results.info.solve_time
            )
        )

    def test_case_update_rho_update_for_different_initial_guess(self):
        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test update rho for different initial guess---"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(rho=1.0e-7)
        assert qp.results.info.rho == 1.0e-7
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

        qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp2.settings.eps_abs = 1.0e-9
        qp2.settings.verbose = False
        qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp2.update(rho=1.0e-7)
        assert qp2.results.info.rho == 1.0e-7
        qp2.solve()
        dua_res = normInf(
            H @ qp2.results.x
            + g
            + A.transpose() @ qp2.results.y
            + C.transpose() @ qp2.results.z
        )
        pri_res = max(
            normInf(A @ qp2.results.x - b),
            normInf(
                np.maximum(C @ qp2.results.x - u, 0)
                + np.minimum(C @ qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp2.results.info.setup_time, qp2.results.info.solve_time
            )
        )

        qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp3.settings.eps_abs = 1.0e-9
        qp3.settings.verbose = False
        qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp3.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp3.solve()
        dua_res = normInf(
            H @ qp3.results.x
            + g
            + A.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A @ qp3.results.x - b),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp3.update(rho=1.0e-7)
        assert qp3.results.info.rho == 1.0e-7
        qp3.solve()
        dua_res = normInf(
            H @ qp3.results.x
            + g
            + A.transpose() @ qp3.results.y
            + C.transpose() @ qp3.results.z
        )
        pri_res = max(
            normInf(A @ qp3.results.x - b),
            normInf(
                np.maximum(C @ qp3.results.x - u, 0)
                + np.minimum(C @ qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp3.results.info.setup_time, qp3.results.info.solve_time
            )
        )

        qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp4.settings.eps_abs = 1.0e-9
        qp4.settings.verbose = False
        qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp4.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp4.solve()
        dua_res = normInf(
            H @ qp4.results.x
            + g
            + A.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A @ qp4.results.x - b),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp4.update(rho=1.0e-7)
        assert qp4.results.info.rho == 1.0e-7
        qp4.solve()
        dua_res = normInf(
            H @ qp4.results.x
            + g
            + A.transpose() @ qp4.results.y
            + C.transpose() @ qp4.results.z
        )
        pri_res = max(
            normInf(A @ qp4.results.x - b),
            normInf(
                np.maximum(C @ qp4.results.x - u, 0)
                + np.minimum(C @ qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp4.results.info.setup_time, qp4.results.info.solve_time
            )
        )

        qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp5.settings.eps_abs = 1.0e-9
        qp5.settings.verbose = False
        qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp5.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
        )
        qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z)
        dua_res = normInf(
            H @ qp5.results.x
            + g
            + A.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A @ qp5.results.x - b),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp5.update(rho=1.0e-7)
        assert qp5.results.info.rho == 1.0e-7
        qp5.solve()
        dua_res = normInf(
            H @ qp5.results.x
            + g
            + A.transpose() @ qp5.results.y
            + C.transpose() @ qp5.results.z
        )
        pri_res = max(
            normInf(A @ qp5.results.x - b),
            normInf(
                np.maximum(C @ qp5.results.x - u, 0)
                + np.minimum(C @ qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp5.results.info.setup_time, qp5.results.info.solve_time
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

        H = spa.csc_matrix(M.dot(M.transpose())).toarray(order="C")
        g = -np.ones((n,))
        A = None
        b = None
        C = spa.csc_matrix(spa.eye(n)).toarray(order="C")
        l = 2.0 * np.ones((n,))
        u = np.full(l.shape, +np.inf)

        qp = proxsuite.proxqp.dense.QP(n, 0, n)
        qp.init(H, g, A, b, C, l, u)
        qp.solve()
        x_theoretically_optimal = np.array([2.0] * 149 + [3.0])

        dua_res = normInf(H @ qp.results.x + g + C.transpose() @ qp.results.z)
        pri_res = normInf(
            np.maximum(C @ qp.results.x - u, 0) + np.minimum(C @ qp.results.x - l, 0)
        )

        assert dua_res <= 1e-3  # default precision of the solver
        assert pri_res <= 1e-3
        assert normInf(x_theoretically_optimal - qp.results.x) <= 1e-3
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, 0, n))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_no_initial_guess(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, no initial guess, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            qp.solve()
            assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
            assert np.abs(rho - qp.results.info.rho) < 1.0e-9
            assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_EQUALITY_CONSTRAINED_INITIAL_GUESS(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, EQUALITY_CONSTRAINED_INITIAL_GUESS, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            qp.solve()
            assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
            assert np.abs(rho - qp.results.info.rho) < 1.0e-9
            assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_COLD_START_WITH_PREVIOUS_RESULT(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, COLD_START_WITH_PREVIOUS_RESULT, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            qp.solve()
            assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
            assert np.abs(rho - qp.results.info.rho) < 1.0e-9
            assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_WARM_START_WITH_PREVIOUS_RESULT(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, WARM_START_WITH_PREVIOUS_RESULT, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            qp.solve()
            assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
            assert np.abs(rho - qp.results.info.rho) < 1.0e-9
            assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_WARM_START_WITH_PREVIOUS_RESULT(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, WARM_START_WITH_PREVIOUS_RESULT, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(mu_eq=1.0e-3, rho=1.0e-6)
        assert np.abs(1.0e-6 - qp.settings.default_rho) < 1.0e-9
        assert np.abs(1.0e-6 - qp.results.info.rho) < 1.0e-9
        assert np.abs(1.0e-3 - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_COLD_START_WITH_PREVIOUS_RESULT(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, COLD_START_WITH_PREVIOUS_RESULT, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(mu_eq=1.0e-3, rho=1.0e-6)
        assert np.abs(1.0e-6 - qp.settings.default_rho) < 1.0e-9
        assert np.abs(1.0e-6 - qp.results.info.rho) < 1.0e-9
        assert np.abs(1.0e-3 - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_EQUALITY_CONSTRAINED_INITIAL_GUESS(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, EQUALITY_CONSTRAINED_INITIAL_GUESS, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(mu_eq=1.0e-3, rho=1.0e-6)
        assert np.abs(1.0e-6 - qp.settings.default_rho) < 1.0e-9
        assert np.abs(1.0e-6 - qp.results.info.rho) < 1.0e-9
        assert np.abs(1.0e-3 - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_NO_INITIAL_GUESS(
        self,
    ):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, NO_INITIAL_GUESS, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.settings.eps_abs = 1.0e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(l),
            np.asfortranarray(u),
            True,
            rho=rho,
            mu_eq=mu_eq,
        )
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        assert np.abs(rho - qp.settings.default_rho) < 1.0e-9
        assert np.abs(rho - qp.results.info.rho) < 1.0e-9
        assert np.abs(mu_eq - qp.settings.default_mu_eq) < 1.0e-9
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        qp.update(mu_eq=1.0e-3, rho=1.0e-6)
        assert np.abs(1.0e-6 - qp.settings.default_rho) < 1.0e-9
        assert np.abs(1.0e-6 - qp.results.info.rho) < 1.0e-9
        assert np.abs(1.0e-3 - qp.settings.default_mu_eq) < 1.0e-9
        qp.solve()
        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_initializing_with_None(self):
        print("------------------------test initialization with Nones")

        H = np.array([[65.0, -22.0, -16.0], [-22.0, 14.0, 7.0], [-16.0, 7.0, 5.0]])
        g = np.array([-13.0, 15.0, 7.0])
        A = None
        b = None
        C = None
        u = None
        l = None

        qp = proxsuite.proxqp.dense.QP(3, 0, 0)
        qp.init(H, g, A, b, C, l, u)
        qp.solve()
        print("optimal x: {}".format(qp.results.x))

        dua_res = normInf(H @ qp.results.x + g)

        assert dua_res <= 1e-3  # default precision of the solver
        print("--n = {} ; n_eq = {} ; n_in = {}".format(3, 0, 0))
        print("dual residual = {} ".format(dua_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_z_ordering_with_box_constraints_interface(self):
        print(
            "------------------------test check ordering of z when there are box constraints"
        )

        n = 50
        n_test = 1000
        eps = 1.0e-9
        # inequality and box constraints case
        for i in range(n_test):
            H, g, A, b, C, u, l, u_box, l_box = generate_mixed_qp_with_box(n, i)
            n_eq = A.shape[0]
            n_in = C.shape[0]

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
            qp.init(H, g, A, b, C, l, u, l_box, u_box)
            qp.settings.eps_abs = eps
            qp.solve()

            # if infeasibility is detected, we relax the tolerance and solve again
            if (
                qp.results.info.status == proxsuite.proxqp.PROXQP_DUAL_INFEASIBLE
                or qp.results.info.status == proxsuite.proxqp.PROXQP_PRIMAL_INFEASIBLE
            ):
                print(f"[{i}] {qp.results.info.status=}, solve again.")
                qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
                qp.init(H, g, A, b, C, l, u, l_box, u_box)
                qp.settings.eps_abs = eps
                qp.settings.eps_primal_inf = 1e-12
                qp.settings.eps_dual_inf = 1e-12
                qp.solve()

            assert qp.results.info.status == proxsuite.proxqp.PROXQP_SOLVED

            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z[:n_in]
                + qp.results.z[n_in:]
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
                normInf(
                    np.maximum(qp.results.x - u_box, 0)
                    + np.minimum(qp.results.x - l_box, 0)
                ),
            )
            assert dua_res <= eps
            assert pri_res <= eps
        # no inequality and box constraints case
        for i in range(n_test):
            H, g, A, b, C, u, l, u_box, l_box = generate_mixed_qp_with_box(n, i)
            n_eq = A.shape[0]
            n_in = 0
            C = np.zeros((n_in, n))
            u = np.zeros(n_in)
            l = np.zeros(n_in)

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
            qp.init(H, g, A, b, C, l, u, l_box, u_box)
            qp.settings.eps_abs = eps
            qp.solve()

            # if infeasibility is detected, we relax the tolerance and solve again
            if (
                qp.results.info.status == proxsuite.proxqp.PROXQP_DUAL_INFEASIBLE
                or qp.results.info.status == proxsuite.proxqp.PROXQP_PRIMAL_INFEASIBLE
            ):
                print(f"[{i}] {qp.results.info.status=}, solve again.")
                qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
                qp.init(H, g, A, b, C, l, u, l_box, u_box)
                qp.settings.eps_abs = eps
                qp.settings.eps_primal_inf = 1e-12
                qp.settings.eps_dual_inf = 1e-12
                qp.solve()

            assert qp.results.info.status == proxsuite.proxqp.PROXQP_SOLVED

            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z[:n_in]
                + qp.results.z[n_in:]
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
                normInf(
                    np.maximum(qp.results.x - u_box, 0)
                    + np.minimum(qp.results.x - l_box, 0)
                ),
            )
            assert dua_res <= eps
            assert pri_res <= eps

        # # no inequality, no equalities and box constraints case
        for i in range(n_test):
            H, g, A, b, C, u, l, u_box, l_box = generate_mixed_qp_with_box(n, i)
            n_eq = 0
            n_in = 0
            C = np.zeros((n_in, n))
            u = np.zeros(n_in)
            l = np.zeros(n_in)
            A = np.zeros((n_eq, n))
            b = np.zeros(n_eq)

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
            qp.init(H, g, A, b, C, l, u, l_box, u_box)
            qp.settings.eps_abs = eps
            qp.solve()

            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z[:n_in]
                + qp.results.z[n_in:]
            )
            pri_res = max(
                normInf(A @ qp.results.x - b),
                normInf(
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                ),
                normInf(
                    np.maximum(qp.results.x - u_box, 0)
                    + np.minimum(qp.results.x - l_box, 0)
                ),
            )
            assert dua_res <= eps
            assert pri_res <= eps
        """
    """

    def test_updates_with_box_constraints_interface(self):
        print(
            "------------------------test check updates work when there are box constraints"
        )
        n = 50
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        eps = 1.0e-9
        n_eq = A.shape[0]
        n_in = C.shape[0]
        u_box = np.ones(n) * 100
        l_box = -np.ones(n) * 100

        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
        qp.init(H, g, A, b, C, l, u, l_box, u_box)
        qp.settings.eps_abs = eps
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z[:n_in]
            + qp.results.z[n_in:]
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
            normInf(
                np.maximum(qp.results.x - u_box, 0)
                + np.minimum(qp.results.x - l_box, 0)
            ),
        )
        assert dua_res <= eps
        assert pri_res <= eps
        u_box += 10
        l_box -= 10

        qp.update(l=l, u=u, u_box=u_box, l_box=l_box)
        qp.solve()

        dua_res = normInf(
            H @ qp.results.x
            + g
            + A.transpose() @ qp.results.y
            + C.transpose() @ qp.results.z[:n_in]
            + qp.results.z[n_in:]
        )
        pri_res = max(
            normInf(A @ qp.results.x - b),
            normInf(
                np.maximum(C @ qp.results.x - u, 0)
                + np.minimum(C @ qp.results.x - l, 0)
            ),
            normInf(
                np.maximum(qp.results.x - u_box, 0)
                + np.minimum(qp.results.x - l_box, 0)
            ),
        )
        assert dua_res <= eps
        assert pri_res <= eps

    def test_dense_infeasibility_solving(
        self,
    ):
        print(
            "------------------------dense random strongly convex qp with inequality constraints, test infeasibility solving"
        )
        n = 20
        for i in range(20):
            H, g, A, b, C, u, l = generate_mixed_qp(n, i)
            b += 10.0  ## create infeasible pbls
            u -= 100.0
            n_eq = A.shape[0]
            n_in = C.shape[0]
            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.settings.eps_abs = 1.0e-5
            qp.settings.eps_primal_inf = 1.0e-4
            qp.settings.verbose = False
            qp.settings.primal_infeasibility_solving = True
            qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
            qp.init(
                H,
                np.asfortranarray(g),
                A,
                np.asfortranarray(b),
                C,
                np.asfortranarray(l),
                np.asfortranarray(u),
            )
            qp.solve()
            dua_res = normInf(
                H @ qp.results.x
                + g
                + A.transpose() @ qp.results.y
                + C.transpose() @ qp.results.z
            )
            ones = A.T @ np.ones(n_eq) + C.T @ np.ones(n_in)

            scaled_eps = normInf(ones) * qp.settings.eps_abs
            pri_res = normInf(
                A.T @ (A @ qp.results.x - b)
                + C.T
                @ (
                    np.maximum(C @ qp.results.x - u, 0)
                    + np.minimum(C @ qp.results.x - l, 0)
                )
            )
            assert dua_res <= qp.settings.eps_abs
            assert pri_res <= scaled_eps

    def test_minimal_eigenvalue_estimation_nonconvex_eigen_option(
        self,
    ):
        print(
            "------------------------dense non convex qp with inequality constraints, estimate minimal eigenvalue with eigen method"
        )
        n = 50
        tol = 1.0e-3
        for i in range(50):
            H, g, A, b, C, u, l = generate_mixed_qp(n, i, -0.01)
            n_eq = A.shape[0]
            n_in = C.shape[0]
            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.settings.verbose = False
            qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
            estimate_minimal_eigen_value = (
                proxsuite.proxqp.dense.estimate_minimal_eigen_value_of_symmetric_matrix(
                    H,
                    proxsuite.proxqp.EigenValueEstimateMethodOption.ExactMethod,
                    1.0e-6,
                    10000,
                )
            )
            vals, _ = spa.linalg.eigs(H, which="SR")
            min_eigenvalue = float(np.min(vals))
            qp.init(
                H,
                np.asfortranarray(g),
                A,
                np.asfortranarray(b),
                C,
                np.asfortranarray(l),
                np.asfortranarray(u),
                manual_minimal_H_eigenvalue=estimate_minimal_eigen_value,
            )
            assert (
                np.abs(min_eigenvalue - qp.results.info.minimal_H_eigenvalue_estimate)
                <= tol
            )

    def test_minimal_eigenvalue_estimation_nonconvex_manual_option(
        self,
    ):
        print(
            "------------------------dense non convex qp with inequality constraints, estimate minimal eigenvalue with manual option"
        )
        n = 50
        tol = 1.0e-3
        for i in range(50):
            H, g, A, b, C, u, l = generate_mixed_qp(n, i, -0.01)
            n_eq = A.shape[0]
            n_in = C.shape[0]
            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.settings.verbose = False
            qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
            vals, _ = spa.linalg.eigs(H, which="SR")
            min_eigenvalue = float(np.min(vals))
            qp.init(
                H,
                np.asfortranarray(g),
                A,
                np.asfortranarray(b),
                C,
                np.asfortranarray(l),
                np.asfortranarray(u),
                manual_minimal_H_eigenvalue=min_eigenvalue,
            )
            assert (
                np.abs(min_eigenvalue - qp.results.info.minimal_H_eigenvalue_estimate)
                <= tol
            )

    def test_minimal_eigenvalue_estimation_nonconvex_power_iter_option(
        self,
    ):
        print(
            "------------------------dense non convex qp with inequality constraints, estimate minimal eigenvalue with power iter option"
        )
        n = 50
        tol = 1.0e-3
        for i in range(50):
            H, g, A, b, C, u, l = generate_mixed_qp(n, i, -0.01)
            n_eq = A.shape[0]
            n_in = C.shape[0]

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.settings.verbose = False
            qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
            estimate_minimal_eigen_value = (
                proxsuite.proxqp.dense.estimate_minimal_eigen_value_of_symmetric_matrix(
                    H,
                    proxsuite.proxqp.EigenValueEstimateMethodOption.PowerIteration,
                    1.0e-6,
                    10000,
                )
            )
            vals, _ = spa.linalg.eigs(H, which="SR")
            min_eigenvalue = float(np.min(vals))
            qp.init(
                H,
                np.asfortranarray(g),
                A,
                np.asfortranarray(b),
                C,
                np.asfortranarray(l),
                np.asfortranarray(u),
                manual_minimal_H_eigenvalue=estimate_minimal_eigen_value,
            )
            assert (
                np.abs(min_eigenvalue - qp.results.info.minimal_H_eigenvalue_estimate)
                <= tol
            )


if __name__ == "__main__":
    unittest.main()
