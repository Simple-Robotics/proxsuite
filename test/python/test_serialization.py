#
# Copyright (c) 2022, INRIA
#
import proxsuite
import numpy as np
import scipy.sparse as spa
import unittest
import pickle


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
    )
    v = np.random.randn(n)  # Fictitious solution
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


def generic_test(object, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(object, f)
    except pickle.PickleError:
        dump_success = False
    else:
        dump_success = True

    assert dump_success

    try:
        with open(filename, "rb") as f:
            loaded_object = pickle.load(f)
    except pickle.PickleError:
        read_success = False
    else:
        read_success = True

    assert read_success
    assert loaded_object == object


class DenseqpWrapperSerialization(unittest.TestCase):
    def test_pickle(self):
        print("------------------------test pickle")
        n = 10
        H, g, A, b, C, u, l = generate_mixed_qp(n)
        n_eq = A.shape[0]
        n_in = C.shape[0]
        rho = 1.0e-7
        mu_eq = 1.0e-4
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
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

        generic_test(qp.model, "qp_model")
        generic_test(qp.settings, "qp_settings")
        generic_test(qp.results, "qp_results")
        generic_test(qp, "qp_wrapper")


if __name__ == "__main__":
    unittest.main()
