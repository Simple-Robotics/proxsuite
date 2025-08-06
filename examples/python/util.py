import numpy as np
import scipy.sparse as spa
import scipy.io as spio

from dataclasses import dataclass


def generate_mixed_qp(n, sparse=False, seed=1, reg=1e-2, dens1=0.075):
    # A function for generating sparse random convex qps

    np.random.seed(seed)
    n_eq = int(n / 4)
    n_in = int(n / 4)
    m = n_eq + n_in

    P = spa.random(
        n, n, density=dens1, data_rvs=np.random.randn, format="csc"
    ).toarray()
    P = (P + P.T) / 2.0

    s = max(np.absolute(np.linalg.eigvals(P)))
    P += (abs(s) + reg) * spa.eye(n)
    P = spa.coo_matrix(P)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    if not sparse:
        A = A.toarray()
        P = P.toarray()
    v = np.random.randn(n)  # Fictitious solution
    _delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P, q, A[:n_eq, :], u[:n_eq], A[n_eq:, :], u[n_eq:], l[n_eq:]


def sparse_positive_definite_rand_not_compressed(dim, rho, p, rng):
    # Inspired from "proxsuite/proxqp/utils/random_qp_problems.hpp"

    H = np.zeros((dim, dim), dtype=np.float64)

    urandom = rng.uniform(size=(dim, dim))
    mask = urandom < (p / 2)
    H[mask] = rng.standard_normal(np.count_nonzero(mask))
    H = 0.5 * (H + H.T)

    eigvals = np.linalg.eigvalsh(H)
    min_eig = eigvals.min()
    H[np.diag_indices(dim)] += rho + abs(min_eig)

    return H


def sparse_matrix_rand_not_compressed(nrows, ncols, p, rng):
    # Inspired from "proxsuite/proxqp/utils/random_qp_problems.hpp"

    mask = rng.uniform(size=(nrows, ncols)) < p

    A = np.zeros((nrows, ncols), dtype=np.float64)
    A[mask] = rng.standard_normal(np.count_nonzero(mask))

    return A


def dense_degenerate_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    # Inspired from "proxsuite/proxqp/utils/random_qp_problems.hpp"

    rng = np.random.default_rng(seed)

    H = sparse_positive_definite_rand_not_compressed(
        dim, strong_convexity_factor, sparsity_factor, rng=rng
    )
    g = rng.standard_normal(dim)

    A = sparse_matrix_rand_not_compressed(n_eq, dim, sparsity_factor, rng=rng)
    C_ = sparse_matrix_rand_not_compressed(n_in, dim, sparsity_factor, rng=rng)
    C = np.vstack([C_, C_])

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    x_sol = rng.standard_normal(dim)
    delta = rng.uniform(size=2 * n_in)

    b = A @ x_sol

    u = C @ x_sol + delta
    l = -1.0e20 * np.ones(2 * n_in)

    return H, g, A, b, C, u, l


@dataclass
class MarosMeszarosQp:
    filename: str
    P: spa.csc_matrix
    q: np.ndarray
    A: spa.csc_matrix
    l: np.ndarray
    u: np.ndarray


@dataclass
class PreprocessedQp:
    H: np.ndarray
    A: np.ndarray
    C: np.ndarray
    g: np.ndarray
    b: np.ndarray
    u: np.ndarray
    l: np.ndarray


def load_qp(filename: str) -> MarosMeszarosQp:
    assert filename.endswith(".mat")
    mat_dict = spio.loadmat(filename)

    P = mat_dict["P"].astype(float).tocsc()
    q = mat_dict["q"].T.flatten().astype(float)
    A = mat_dict["A"].astype(float).tocsc()
    l = mat_dict["l"].T.flatten().astype(float)
    u = mat_dict["u"].T.flatten().astype(float)

    return MarosMeszarosQp(filename=filename, P=P, q=q, A=A, l=l, u=u)


def preprocess_qp(qp: MarosMeszarosQp) -> PreprocessedQp:
    eq = np.isclose(qp.l, qp.u, atol=1e-4)

    n = qp.P.shape[0]
    n_eq = np.sum(eq)
    n_in = len(eq) - n_eq

    A = np.zeros((n_eq, n))
    b = np.zeros(n_eq)

    C = np.zeros((n_in, n))
    u = np.zeros(n_in)
    l = np.zeros(n_in)

    eq_idx = 0
    in_idx = 0

    for i in range(len(eq)):
        if eq[i]:
            A[eq_idx, :] = qp.A[i, :].toarray().flatten()
            b[eq_idx] = qp.l[i]
            eq_idx += 1
        else:
            C[in_idx, :] = qp.A[i, :].toarray().flatten()
            l[in_idx] = qp.l[i]
            u[in_idx] = qp.u[i]
            in_idx += 1

    return PreprocessedQp(H=qp.P.toarray(), A=A, C=C, g=qp.q, b=b, u=u, l=l)
