#
# Copyright (c) 2022, INRIA
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
    Generate problem in QP format
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
    print("sparsity of P : {}".format((P.nnz) / (n**2)))
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    v = np.random.randn(n)  # Fictitious solution
    delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P, q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


class DenseQpWrapper(unittest.TestCase):

    # TESTS OF GENERAL METHODS OF THE API

    def test_case_update_rho(self):
        print(
            "------------------------sparse random strongly convex qp with equality and inequality constraints: test update rho"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.init(
            H=H,
            g=np.asfortranarray(g),
            A=A,
            b=np.asfortranarray(b),
            C=C,
            u=np.asfortranarray(u),
            l=np.asfortranarray(l),
            rho=1.0e-7,
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            mu_eq=1.0e-2,
            mu_in=1.0e-3,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            False,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        g = np.random.randn(n)
        H *= 2.0  # too keep same sparsity structure
        Qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )

        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )
        x_wm = np.random.randn(n)
        y_wm = np.random.randn(n_eq)
        z_wm = np.random.randn(n_in)
        Qp.solve(x_wm, y_wm, z_wm)

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1e-9
        assert dua_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )

        x = Qp.results.x
        y = Qp.results.y
        z = Qp.results.z
        Qp2.solve(x, y, z)

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp2".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
        )
        x = Qp.results.x
        y = Qp.results.y
        z = Qp.results.z
        Qp2.solve(x, y, z)

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp2".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            False,
        )
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} after warm starting with Qp2".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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

        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print("--n = {} ; n_eq = {} ; n_in = {} with Qp".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.update(update_preconditioner=True)
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} with Qp after update".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            False,
        )
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print("--n = {} ; n_eq = {} ; n_in = {} with Qp2".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2.update(update_preconditioner=False)
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print(
            "--n = {} ; n_eq = {} ; n_in = {} with Qp2 after update".format(
                n, n_eq, n_in
            )
        )
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve(np.random.randn(n), np.random.randn(n_eq), np.random.randn(n_in))

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert pri_res <= 1.0e-9
        assert dua_res <= 1.0e-9
        print("--n = {} ; n_eq = {} ; n_in = {} with Qp".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        Qp.init(H, g, A, b, C, u, l)
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START

        Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        Qp.init(H, g, A, b, C, u, l)
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.init(H, g, A, b, C, u, l)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        Qp2.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve with new QP object")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp2.results.info.setup_time, Qp2.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            update_preconditioner,
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        Qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            update_preconditioner,
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        Qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            update_preconditioner,
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        Qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            update_preconditioner,
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )

        H *= 2  # keep same sparsity structure
        g = np.random.randn(n)
        Qp.update(H=H, g=np.asfortranarray(g), update_preconditioner=True)
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()

        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START

        H *= 2.0  # keep same sparsity structure
        g = np.random.randn(n)
        update_preconditioner = True
        Qp.update(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            update_preconditioner,
        )
        Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Second solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Third solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z)
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("Fourth solve ")
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=1.0e-7,
        )
        Qp.solve()
        assert Qp.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        Qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=1.0e-7,
        )
        Qp2.solve()
        assert Qp2.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp2.results.info.setup_time, Qp2.results.info.solve_time
            )
        )

        Qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp3.settings.eps_abs = 1.0e-9
        Qp3.settings.verbose = False
        Qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp3.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=1.0e-7,
        )
        Qp3.solve()
        assert Qp.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ Qp3.results.x
            + g
            + A.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp3.results.info.setup_time, Qp3.results.info.solve_time
            )
        )

        Qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp4.settings.eps_abs = 1.0e-9
        Qp4.settings.verbose = False
        Qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        Qp4.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=1.0e-7,
        )
        Qp4.solve()
        assert Qp4.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ Qp4.results.x
            + g
            + A.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp4.results.info.setup_time, Qp4.results.info.solve_time
            )
        )

        Qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp5.settings.eps_abs = 1.0e-9
        Qp5.settings.verbose = False
        Qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp5.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=1.0e-7,
        )
        Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z)
        assert Qp.results.info.rho == 1.0e-7
        dua_res = normInf(
            H @ Qp5.results.x
            + g
            + A.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp5.results.info.setup_time, Qp5.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()
        g = np.random.randn(n)
        dua_res = normInf(
            H @ Qp.results.x
            + g_old
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(g=g)
        assert normInf(Qp.model.g - g) <= 1.0e-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        Qp2.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g_old
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp2.update(g=g)
        assert normInf(Qp.model.g - g) <= 1.0e-9
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp2.results.info.setup_time, Qp2.results.info.solve_time
            )
        )

        Qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp3.settings.eps_abs = 1.0e-9
        Qp3.settings.verbose = False
        Qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp3.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp3.solve()
        dua_res = normInf(
            H @ Qp3.results.x
            + g_old
            + A.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp3.update(g=g)
        assert normInf(Qp.model.g - g) <= 1.0e-9
        Qp3.solve()
        dua_res = normInf(
            H @ Qp3.results.x
            + g
            + A.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp3.results.info.setup_time, Qp3.results.info.solve_time
            )
        )

        Qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp4.settings.eps_abs = 1.0e-9
        Qp4.settings.verbose = False
        Qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        Qp4.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp4.solve()
        dua_res = normInf(
            H @ Qp4.results.x
            + g_old
            + A.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp4.update(g=g)
        assert normInf(Qp.model.g - g) <= 1.0e-9
        Qp4.solve()
        dua_res = normInf(
            H @ Qp4.results.x
            + g
            + A.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp4.results.info.setup_time, Qp4.results.info.solve_time
            )
        )

        Qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp5.settings.eps_abs = 1.0e-9
        Qp5.settings.verbose = False
        Qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp5.init(
            H,
            np.asfortranarray(g_old),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z)
        dua_res = normInf(
            H @ Qp5.results.x
            + g_old
            + A.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp5.update(g=g)
        assert normInf(Qp.model.g - g) <= 1.0e-9
        Qp5.solve()
        dua_res = normInf(
            H @ Qp5.results.x
            + g
            + A.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp5.results.info.setup_time, Qp5.results.info.solve_time
            )
        )

    def test_case_update_A_for_different_initial_guess(self):

        print(
            "---testing sparse random strongly convex qp with equality and inequality constraints: test update A for different initial guess---"
        )
        n = 10
        H, g, A_old, b, C, u, l = generate_mixed_qp(n)
        n_eq = A_old.shape[0]
        n_in = C.shape[0]
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()
        A = spa.random(n_eq, n, density=0.15, data_rvs=np.random.randn, format="csc")
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A_old.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A_old @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(A=A)
        assert normInf(Qp.model.A - A) <= 1.0e-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        Qp2.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A_old.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A_old @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp2.update(A=A)
        assert normInf(Qp.model.A - A) <= 1.0e-9
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp2.results.info.setup_time, Qp2.results.info.solve_time
            )
        )

        Qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp3.settings.eps_abs = 1.0e-9
        Qp3.settings.verbose = False
        Qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp3.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp3.solve()
        dua_res = normInf(
            H @ Qp3.results.x
            + g
            + A_old.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A_old @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp3.update(A=A)
        assert normInf(Qp.model.A - A) <= 1.0e-9
        Qp3.solve()
        dua_res = normInf(
            H @ Qp3.results.x
            + g
            + A.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp3.results.info.setup_time, Qp3.results.info.solve_time
            )
        )

        Qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp4.settings.eps_abs = 1.0e-9
        Qp4.settings.verbose = False
        Qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        Qp4.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp4.solve()
        dua_res = normInf(
            H @ Qp4.results.x
            + g
            + A_old.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A_old @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp4.update(A=A)
        assert normInf(Qp.model.A - A) <= 1.0e-9
        Qp4.solve()
        dua_res = normInf(
            H @ Qp4.results.x
            + g
            + A.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp4.results.info.setup_time, Qp4.results.info.solve_time
            )
        )

        Qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp5.settings.eps_abs = 1.0e-9
        Qp5.settings.verbose = False
        Qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp5.init(
            H,
            np.asfortranarray(g),
            A_old,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z)
        dua_res = normInf(
            H @ Qp5.results.x
            + g
            + A_old.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A_old @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp5.update(A=A)
        assert normInf(Qp.model.A - A) <= 1.0e-9
        Qp5.solve()
        dua_res = normInf(
            H @ Qp5.results.x
            + g
            + A.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp5.results.info.setup_time, Qp5.results.info.solve_time
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
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(rho=1.0e-7)
        assert Qp.results.info.rho == 1.0e-7
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )

        Qp2 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp2.settings.eps_abs = 1.0e-9
        Qp2.settings.verbose = False
        Qp2.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        Qp2.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp2.update(rho=1.0e-7)
        assert Qp2.results.info.rho == 1.0e-7
        Qp2.solve()
        dua_res = normInf(
            H @ Qp2.results.x
            + g
            + A.transpose() @ Qp2.results.y
            + C.transpose() @ Qp2.results.z
        )
        pri_res = max(
            normInf(A @ Qp2.results.x - b),
            normInf(
                np.maximum(C @ Qp2.results.x - u, 0)
                + np.minimum(C @ Qp2.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp2.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp2.results.info.setup_time, Qp2.results.info.solve_time
            )
        )

        Qp3 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp3.settings.eps_abs = 1.0e-9
        Qp3.settings.verbose = False
        Qp3.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        )
        Qp3.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp3.solve()
        dua_res = normInf(
            H @ Qp3.results.x
            + g
            + A.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp3.update(rho=1.0e-7)
        assert Qp3.results.info.rho == 1.0e-7
        Qp3.solve()
        dua_res = normInf(
            H @ Qp3.results.x
            + g
            + A.transpose() @ Qp3.results.y
            + C.transpose() @ Qp3.results.z
        )
        pri_res = max(
            normInf(A @ Qp3.results.x - b),
            normInf(
                np.maximum(C @ Qp3.results.x - u, 0)
                + np.minimum(C @ Qp3.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp3.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp3.results.info.setup_time, Qp3.results.info.solve_time
            )
        )

        Qp4 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp4.settings.eps_abs = 1.0e-9
        Qp4.settings.verbose = False
        Qp4.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        )
        Qp4.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp4.solve()
        dua_res = normInf(
            H @ Qp4.results.x
            + g
            + A.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp4.update(rho=1.0e-7)
        assert Qp4.results.info.rho == 1.0e-7
        Qp4.solve()
        dua_res = normInf(
            H @ Qp4.results.x
            + g
            + A.transpose() @ Qp4.results.y
            + C.transpose() @ Qp4.results.z
        )
        pri_res = max(
            normInf(A @ Qp4.results.x - b),
            normInf(
                np.maximum(C @ Qp4.results.x - u, 0)
                + np.minimum(C @ Qp4.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp4.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp4.results.info.setup_time, Qp4.results.info.solve_time
            )
        )

        Qp5 = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp5.settings.eps_abs = 1.0e-9
        Qp5.settings.verbose = False
        Qp5.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp5.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
        )
        Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z)
        dua_res = normInf(
            H @ Qp5.results.x
            + g
            + A.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp5.update(rho=1.0e-7)
        assert Qp5.results.info.rho == 1.0e-7
        Qp5.solve()
        dua_res = normInf(
            H @ Qp5.results.x
            + g
            + A.transpose() @ Qp5.results.y
            + C.transpose() @ Qp5.results.z
        )
        pri_res = max(
            normInf(A @ Qp5.results.x - b),
            normInf(
                np.maximum(C @ Qp5.results.x - u, 0)
                + np.minimum(C @ Qp5.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, n_eq, n_in))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp5.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp5.results.info.setup_time, Qp5.results.info.solve_time
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

        H = spa.csc_matrix(M.dot(M.transpose()))
        g = -np.ones((n,))
        A = None
        b = None
        C = spa.csc_matrix(spa.eye(n))
        l = 2.0 * np.ones((n,))
        u = np.full(l.shape, +np.infty)

        Qp = proxsuite.proxqp.dense.QP(n, 0, n)
        Qp.init(H, g, A, b, C, u, l)
        Qp.solve()
        x_theoretically_optimal = np.array([2.0] * 149 + [3.0])

        dua_res = normInf(H @ Qp.results.x + g + C.transpose() @ Qp.results.z)
        pri_res = normInf(
            np.maximum(C @ Qp.results.x - u, 0) + np.minimum(C @ Qp.results.x - l, 0)
        )

        assert dua_res <= 1e-3  # default precision of the solver
        assert pri_res <= 1e-3
        assert normInf(x_theoretically_optimal - Qp.results.x) <= 1e-3
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, 0, n))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )



    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_no_initial_guess(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, no initial guess, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            Qp.solve()
            assert np.abs(rho - Qp.settings.default_rho) <1.E-9
            assert np.abs(rho - Qp.results.info.rho) <1.E-9
            assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
            dua_res = normInf(
                H @ Qp.results.x
                + g
                + A.transpose() @ Qp.results.y
                + C.transpose() @ Qp.results.z
            )
            pri_res = max(
                normInf(A @ Qp.results.x - b),
                normInf(
                    np.maximum(C @ Qp.results.x - u, 0)
                    + np.minimum(C @ Qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_EQUALITY_CONSTRAINED_INITIAL_GUESS(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, EQUALITY_CONSTRAINED_INITIAL_GUESS, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            Qp.solve()
            assert np.abs(rho - Qp.settings.default_rho) <1.E-9
            assert np.abs(rho - Qp.results.info.rho) <1.E-9
            assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
            dua_res = normInf(
                H @ Qp.results.x
                + g
                + A.transpose() @ Qp.results.y
                + C.transpose() @ Qp.results.z
            )
            pri_res = max(
                normInf(A @ Qp.results.x - b),
                normInf(
                    np.maximum(C @ Qp.results.x - u, 0)
                    + np.minimum(C @ Qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_COLD_START_WITH_PREVIOUS_RESULT(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, COLD_START_WITH_PREVIOUS_RESULT, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            Qp.solve()
            assert np.abs(rho - Qp.settings.default_rho) <1.E-9
            assert np.abs(rho - Qp.results.info.rho) <1.E-9
            assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
            dua_res = normInf(
                H @ Qp.results.x
                + g
                + A.transpose() @ Qp.results.y
                + C.transpose() @ Qp.results.z
            )
            pri_res = max(
                normInf(A @ Qp.results.x - b),
                normInf(
                    np.maximum(C @ Qp.results.x - u, 0)
                    + np.minimum(C @ Qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9

    def test_sparse_problem_multiple_solve_with_default_rho_mu_eq_and_WARM_START_WITH_PREVIOUS_RESULT(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, WARM_START_WITH_PREVIOUS_RESULT, multiple solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        for i in range(10):
            Qp.solve()
            assert np.abs(rho - Qp.settings.default_rho) <1.E-9
            assert np.abs(rho - Qp.results.info.rho) <1.E-9
            assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
            dua_res = normInf(
                H @ Qp.results.x
                + g
                + A.transpose() @ Qp.results.y
                + C.transpose() @ Qp.results.z
            )
            pri_res = max(
                normInf(A @ Qp.results.x - b),
                normInf(
                    np.maximum(C @ Qp.results.x - u, 0)
                    + np.minimum(C @ Qp.results.x - l, 0)
                ),
            )
            assert dua_res <= 1e-9
            assert pri_res <= 1e-9


    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_WARM_START_WITH_PREVIOUS_RESULT(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, WARM_START_WITH_PREVIOUS_RESULT, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(mu_eq = 1.E-3,rho=1.E-6)
        assert np.abs(1.E-6 - Qp.settings.default_rho) <1.E-9
        assert np.abs(1.E-6 - Qp.results.info.rho) <1.E-9
        assert np.abs(1.E-3 - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_COLD_START_WITH_PREVIOUS_RESULT(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, COLD_START_WITH_PREVIOUS_RESULT, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.COLD_START_WITH_PREVIOUS_RESULT
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(mu_eq = 1.E-3,rho=1.E-6)
        assert np.abs(1.E-6 - Qp.settings.default_rho) <1.E-9
        assert np.abs(1.E-6 - Qp.results.info.rho) <1.E-9
        assert np.abs(1.E-3 - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_EQUALITY_CONSTRAINED_INITIAL_GUESS(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, EQUALITY_CONSTRAINED_INITIAL_GUESS, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.EQUALITY_CONSTRAINED_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(mu_eq = 1.E-3,rho=1.E-6)
        assert np.abs(1.E-6 - Qp.settings.default_rho) <1.E-9
        assert np.abs(1.E-6 - Qp.results.info.rho) <1.E-9
        assert np.abs(1.E-3 - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9

    def test_sparse_problem_update_and_solve_with_default_rho_mu_eq_and_NO_INITIAL_GUESS(self):
        print(
            "------------------------sparse random strongly convex qp with inequality constraints, NO_INITIAL_GUESS, update + solve and default rho and mu_eq"
        )
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.E-7
        mu_eq = 1.E-4
        Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        Qp.settings.eps_abs = 1.0e-9
        Qp.settings.verbose = False
        Qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        Qp.init(
            H,
            np.asfortranarray(g),
            A,
            np.asfortranarray(b),
            C,
            np.asfortranarray(u),
            np.asfortranarray(l),
            True,
            rho=rho,
            mu_eq=mu_eq
        )
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        assert np.abs(rho - Qp.settings.default_rho) <1.E-9
        assert np.abs(rho - Qp.results.info.rho) <1.E-9
        assert np.abs(mu_eq - Qp.settings.default_mu_eq) <1.E-9
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
            ),
        )
        assert dua_res <= 1e-9
        assert pri_res <= 1e-9
        Qp.update(mu_eq = 1.E-3,rho=1.E-6)
        assert np.abs(1.E-6 - Qp.settings.default_rho) <1.E-9
        assert np.abs(1.E-6 - Qp.results.info.rho) <1.E-9
        assert np.abs(1.E-3 - Qp.settings.default_mu_eq) <1.E-9
        Qp.solve()
        dua_res = normInf(
            H @ Qp.results.x
            + g
            + A.transpose() @ Qp.results.y
            + C.transpose() @ Qp.results.z
        )
        pri_res = max(
            normInf(A @ Qp.results.x - b),
            normInf(
                np.maximum(C @ Qp.results.x - u, 0)
                + np.minimum(C @ Qp.results.x - l, 0)
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

        Qp = proxsuite.proxqp.dense.QP(3, 0, 0)
        Qp.init(H, g, A, b, C, u, l)
        Qp.solve()
        print("optimal x: {}".format(Qp.results.x))

        dua_res = normInf(H @ Qp.results.x + g)

        assert dua_res <= 1e-3  # default precision of the solver
        print("--n = {} ; n_eq = {} ; n_in = {}".format(3, 0, 0))
        print("dual residual = {} ".format(dua_res))
        print("total number of iteration: {}".format(Qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                Qp.results.info.setup_time, Qp.results.info.solve_time
            )
        )


if __name__ == "__main__":
    unittest.main()
