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


def generate_invertible_symmetric_matrix(n, seed=1):
    """
    Generate invertible symmetric matrix
    """
    np.random.seed(seed)

    P = spa.random(n, n, density=0.05, data_rvs=np.random.randn, format="csc").toarray()
    a = P @ P.T
    a += np.eye(n)
    a = spa.csc_matrix(a)

    return a


def apply_pt(a, pt):
    # pt is a permutation matrix in vector format
    # return a * pt
    n = a.shape[0]
    res = np.copy(a)
    for i in range(n):
        res[:, pt[i]] = a[:, i]
    return res


def apply_p(p, a):
    # p is a permutation matrix in vector format
    # return p * a
    n = a.shape[0]
    res = np.copy(a)
    for i in range(n):
        res[i, :] = a[p[i], :]
    return res


class SparseLdltApi(unittest.TestCase):

    # TESTS SPARSE LDLT API

    def test_reconstruct_factorized_matrix(self):
        print(
            "------------------------ test reconstruction method of a factorized matrix"
        )

        for n in range(10, 500, 10):
            for i in range(10):
                # print("n : {} ; i : {}".format(n,i))
                a = generate_invertible_symmetric_matrix(n, i)
                ldl = proxsuite.linalg.sparse.SparseLDLT(n)
                ldl.factorize(a)
                a_ = ldl.reconstruct_factorized_matrix()
                err_matrix = a.toarray() - a_
                err = np.linalg.norm(err_matrix, np.inf)
                # print("---err : {}".format(err))
                assert err <= 1e-10

    def test_reconstruct_factorized_matrix_from_l_d_lt_and_permutation_matrices(self):
        print(
            "------------------------test reconstruction of the factorized matrix from l, d, lt and permutation matrices"
        )
        for n in range(10, 500, 10):
            for i in range(10):
                # print("n : {} ; i : {}".format(n,i))
                a = generate_invertible_symmetric_matrix(n, i)
                ldl = proxsuite.linalg.sparse.SparseLDLT(n)
                ldl.factorize(a)
                p = ldl.p()  # this is a vector
                pt = ldl.pt()  # this is a vector
                l = ldl.l()
                lt = ldl.lt()
                d = ldl.d()
                ldl_permuted = l @ d @ lt
                pt_ldl = apply_p(p, ldl_permuted)
                pt_ldl_p = apply_pt(pt_ldl, pt)
                err_matrix = a.toarray() - pt_ldl_p
                err = np.linalg.norm(err_matrix, np.inf)
                # print("---err : {}".format(err))
                assert err <= 1.0e-10

    def test_solve_linear_system(self):
        print("------------------------test linear system solving")
        for n in range(10, 500, 10):
            for i in range(10):
                # print("n : {} ; i : {}".format(n,i))
                a = generate_invertible_symmetric_matrix(n, i)
                ldl = proxsuite.linalg.sparse.SparseLDLT(n)
                ldl.factorize(a)
                rhs = np.random.randn(n)
                sol = np.copy(rhs)
                ldl.solve_in_place(sol)
                err = np.linalg.norm(a @ sol - rhs, np.inf)
                # print("---err : {}".format(err))
                assert err <= 1.0e-10


if __name__ == "__main__":
    unittest.main()
