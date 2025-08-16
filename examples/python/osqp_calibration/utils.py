import proxsuite

import numpy as np
import numpy.linalg as la
import scipy.sparse as spa
import scipy.io as spio

from dataclasses import dataclass


def infty_norm(vec: np.ndarray):
    return la.norm(vec, np.inf, axis=0)


def status_to_string(status, solver):
    if solver == "proxsuite":
        if status == proxsuite.osqp.PROXQP_SOLVED:
            return "Solved"
        elif status == proxsuite.osqp.PROXQP_MAX_ITER_REACHED:
            return "Maximum number of iterations reached"
        elif status == proxsuite.osqp.PROXQP_PRIMAL_INFEASIBLE:
            return "Primal infeasible"
        elif status == proxsuite.osqp.PROXQP_DUAL_INFEASIBLE:
            return "Dual infeasible"
        elif status == proxsuite.osqp.PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE:
            return "Solved closest primal feasible"
        elif status == proxsuite.osqp.PROXQP_NOT_RUN:
            return "Solver not run"

    elif solver == "source":
        if status == "solved":
            return "Solved"
        elif status == "maximum iterations reached":
            return "Maximum number of iterations reached"
        elif status == "primal infeasible":
            return "Primal infeasible"
        elif status == "dual infeasible":
            return "Dual infeasible"

    else:
        print("solver argument must be proxsuite or source")


def status_polish_to_string(status, solver):
    if solver == "proxsuite":
        if status == proxsuite.osqp.POLISH_SUCCEEDED:
            return "Polishing: succeed"
        elif status == proxsuite.osqp.POLISH_FAILED:
            return "Polishing: failed"
        elif status == proxsuite.osqp.POLISH_NOT_RUN:
            return "Polishing: not run"
        elif status == proxsuite.osqp.POLISH_NO_ACTIVE_SET_FOUND:
            return "Polishing: no active set found"

    elif solver == "source":
        if status == 1:
            return "Polishing: succeed"
        elif status == -1:
            return "Polishing: failed"
        elif status == 0:
            return "Polishing: not run"
        elif status == 2:
            return "Polishing: no active set found"

    else:
        print("solver argument must be proxsuite or source")


def sparse_positive_definite_rand_not_compressed(dim, rho, p, rng):
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"

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
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"

    mask = rng.uniform(size=(nrows, ncols)) < p

    A = np.zeros((nrows, ncols), dtype=np.float64)
    A[mask] = rng.standard_normal(np.count_nonzero(mask))

    return A


def unconstrained_qp(
    dim, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"

    rng = np.random.default_rng(seed)

    H = sparse_positive_definite_rand_not_compressed(
        dim, strong_convexity_factor, sparsity_factor, rng=rng
    )
    g = rng.standard_normal(dim)

    A = sparse_matrix_rand_not_compressed(0, dim, sparsity_factor, rng=rng)
    C = sparse_matrix_rand_not_compressed(0, dim, sparsity_factor, rng=rng)

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    b = rng.standard_normal(0)
    u = rng.standard_normal(0)
    l = rng.standard_normal(0)

    return H, g, A, b, C, u, l


def strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"

    rng = np.random.default_rng(seed)

    H = sparse_positive_definite_rand_not_compressed(
        dim, strong_convexity_factor, sparsity_factor, rng=rng
    )
    g = rng.standard_normal(dim)

    A = sparse_matrix_rand_not_compressed(n_eq, dim, sparsity_factor, rng=rng)
    C = sparse_matrix_rand_not_compressed(n_in, dim, sparsity_factor, rng=rng)

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    x_sol = rng.standard_normal(dim)
    delta = rng.uniform(size=n_in)

    b = A @ x_sol

    u = C @ x_sol + delta
    l = -1.0e20 * np.ones(n_in)

    return H, g, A, b, C, u, l


def not_strongly_convex_qp(dim, n_eq, n_in, sparsity_factor, sparse=False, seed=1):
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"

    rng = np.random.default_rng(seed)

    H = sparse_positive_definite_rand_not_compressed(dim, 0, sparsity_factor, rng=rng)
    A = sparse_matrix_rand_not_compressed(n_eq, dim, sparsity_factor, rng=rng)
    C = sparse_matrix_rand_not_compressed(n_in, dim, sparsity_factor, rng=rng)

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    x_sol = rng.standard_normal(dim)
    y_sol = rng.standard_normal(n_eq)
    z_sol = rng.standard_normal(n_in)
    delta = rng.uniform(size=n_in)

    Cx = C @ x_sol
    u = Cx + delta
    l = Cx - delta
    b = A @ x_sol

    g = -(H @ x_sol + A.T @ y_sol + C.T @ z_sol)

    return H, g, A, b, C, u, l


def degenerate_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"

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


def box_constrained_qp(
    dim, n_eq, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    # Inspired from "proxsuite/common/utils/random_qp_problems.hpp"
    # Note: n_in is not in argument, as C must be square with size dim

    rng = np.random.default_rng(seed)

    H = sparse_positive_definite_rand_not_compressed(
        dim, strong_convexity_factor, sparsity_factor, rng=rng
    )
    g = rng.standard_normal(dim)

    A = sparse_matrix_rand_not_compressed(n_eq, dim, sparsity_factor, rng=rng)
    C = np.ones((dim, dim))

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    x_sol = rng.standard_normal(dim)
    delta = rng.uniform(size=dim)

    b = A @ x_sol

    u = C @ x_sol + delta
    l = C @ x_sol - delta

    return H, g, A, b, C, u, l


def primal_infeasible_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    rng = np.random.default_rng(seed)

    H = sparse_positive_definite_rand_not_compressed(
        dim, strong_convexity_factor, sparsity_factor, rng=rng
    )
    g = rng.standard_normal(dim)

    A = sparse_matrix_rand_not_compressed(n_eq, dim, sparsity_factor, rng=rng)
    C = sparse_matrix_rand_not_compressed(n_in, dim, sparsity_factor, rng=rng)

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    x_sol = rng.standard_normal(dim)
    delta = rng.uniform(size=n_in)

    b = A @ x_sol

    u = C @ x_sol + delta
    l = -1.0e20 * np.ones(n_in)

    n_cont = n_in // 2
    for idx in range(n_cont):
        i = 2 * idx
        j = 2 * idx + 1
        if j < n_in:
            C[j] = -C[i]
            u[i] = rng.uniform(1.0, 3.0)
            u[j] = -rng.uniform(u[i] + 0.5, u[i] + 5.0)

    return H, g, A, b, C, u, l


def dual_infeasible_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor=1e-2, sparse=False, seed=1
):
    rng = np.random.default_rng(seed)

    H = np.zeros((dim, dim))
    g = np.ones(dim)

    A = sparse_matrix_rand_not_compressed(n_eq, dim, sparsity_factor, rng=rng)
    C = sparse_matrix_rand_not_compressed(n_in, dim, sparsity_factor, rng=rng)

    if sparse:
        H = spa.csc_matrix(H)
        A = spa.csc_matrix(A)
        C = spa.csc_matrix(C)

    x_sol = rng.standard_normal(dim)
    delta = rng.uniform(size=n_in)

    b = A @ x_sol

    u = C @ x_sol + delta
    l = -1.0e20 * np.ones(n_in)

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
