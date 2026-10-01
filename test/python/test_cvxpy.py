#
# Copyright (c) 2022, INRIA
#

import proxsuite
import numpy as np
import unittest


def normInf(x):
    if x.shape[0] == 0:
        return 0.0
    else:
        return np.linalg.norm(x, np.inf)


class CvxpyTest(unittest.TestCase):
    def test_trigger_infeasibility_with_exact_solution_known(self):
        print(
            "------------------------ test if infeasibility is triggered even though exact solution known"
        )

        n = 3
        H = np.array([[13.0, 12.0, -2.0], [12.0, 17.0, 6.0], [-2.0, 6.0, 12.0]])
        g = np.array([-22.0, -14.5, 13.0])
        A = None
        b = None
        C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        l = -np.ones((n))
        u = np.ones(n)

        qp = proxsuite.proxqp.dense.QP(n, 0, n)
        qp.init(H, g, A, b, C, l, u)
        qp.settings.verbose = True
        qp.solve()
        x_sol = np.array([1, 0.5, -1])

        dua_res = normInf(H @ qp.results.x + g + C.transpose() @ qp.results.z)
        pri_res = normInf(
            np.maximum(C @ qp.results.x - u, 0) + np.minimum(C @ qp.results.x - l, 0)
        )
        assert qp.results.info.status.name == "PROXQP_SOLVED"

        assert dua_res <= 1e-3  # default precision of the solver
        assert pri_res <= 1e-3
        assert normInf(x_sol - qp.results.x) <= 1e-3
        print("--n = {} ; n_eq = {} ; n_in = {}".format(n, 0, n))
        print("dual residual = {} ; primal residual = {}".format(dua_res, pri_res))
        print("total number of iteration: {}".format(qp.results.info.iter))
        print(
            "setup timing = {} ; solve time = {}".format(
                qp.results.info.setup_time, qp.results.info.solve_time
            )
        )

    def test_one_dim_with_exact_solution_known(self):
        print("------------------------ test_one_dim_with_exact_solution_known")
        n = 1
        H = np.array([[20.0]])
        g = np.array([-10.0])
        A = None
        b = None
        C = np.array([[1.0]])
        l = 0 * np.ones((n))
        u = np.ones(n)

        qp = proxsuite.proxqp.dense.QP(n, 0, n)
        qp.init(H, g, A, b, C, l, u)
        qp.settings.verbose = True
        qp.settings.eps_abs = 1e-8
        qp.solve()

        x_sol = 0.5
        assert (x_sol - qp.results.x) <= 1e-4


if __name__ == "__main__":
    unittest.main()
