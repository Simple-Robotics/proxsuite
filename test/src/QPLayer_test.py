from typing import Optional, Tuple, Union
import unittest

import cvxpy as cp
import numpy as np
import proxsuite
import proxsuite.torch.qplayer as qplayer
import scipy.sparse as sp
import torch


def solve_single_qp_numpy(
    H: np.ndarray,
    g: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    C: Optional[np.ndarray],
    l: Optional[np.ndarray],
    u: Optional[np.ndarray],
) -> np.ndarray:
    """Solve a single QP problem using proxsuite numpy backend.

    Args:
        H: Quadratic cost matrix (Hessian)
        g: Linear cost vector
        A: Equality constraint matrix
        b: Equality constraint vector
        C: Inequality constraint matrix
        l: Lower bounds for inequality constraints
        u: Upper bounds for inequality constraints

    Returns:
        Optimal solution vector

    Raises:
        AssertionError: If the QP problem is not feasible
    """
    assert is_qp_feasible(H, g, A, b, C, l, u), "QP problem is not feasible"

    results = proxsuite.proxqp.dense.solve(
        H=H,
        g=np.asfortranarray(g),
        A=A,
        b=np.asfortranarray(b) if b is not None else None,
        C=C,
        l=np.asfortranarray(l) if l is not None else None,
        u=np.asfortranarray(u) if u is not None else None,
        eps_abs=1e-5,
        eps_rel=0,
        verbose=False,
    )
    return results.x


def solve_single_qp_torch_feasible(
    P: torch.Tensor,
    q: torch.Tensor,
    A: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    G: Optional[torch.Tensor],
    lb: Optional[torch.Tensor],
    ub: Optional[torch.Tensor],
) -> torch.Tensor:
    """Solve a single QP problem using torch QPFunction with structural feasibility.

    Args:
        P: Quadratic cost matrix
        q: Linear cost vector
        A: Equality constraint matrix
        b: Equality constraint vector
        G: Inequality constraint matrix
        lb: Lower bounds for inequality constraints
        ub: Upper bounds for inequality constraints

    Returns:
        Optimal solution tensor
    """
    eps = 1e-5
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=True)
    output_torch = function(P, q, A, b, G, lb, ub)
    return output_torch[0]


def solve_single_qp_torch_non_structural_feasible(
    P: torch.Tensor,
    q: torch.Tensor,
    A: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    G: Optional[torch.Tensor],
    lb: Optional[torch.Tensor],
    ub: Optional[torch.Tensor],
) -> torch.Tensor:
    """Solve a single QP problem using torch QPFunction without structural feasibility.

    Args:
        P: Quadratic cost matrix
        q: Linear cost vector
        A: Equality constraint matrix
        b: Equality constraint vector
        G: Inequality constraint matrix
        lb: Lower bounds for inequality constraints
        ub: Upper bounds for inequality constraints

    Returns:
        Optimal solution tensor
    """
    eps = 1e-5
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=False)
    output_torch = function(P, q, A, b, G, lb, ub)
    return output_torch[0]


def batch_generate_qps(
    generate_qp_func: callable,
    n_batch: int,
    qp_size: int = 10,
) -> Tuple[np.ndarray, ...]:
    """Generate a batch of QP problems for testing.

    Args:
        generate_qp_func: Function to generate a single QP problem
        n_batch: Number of QP problems to generate
        qp_size: Size of each QP problem

    Returns:
        Tuple of stacked numpy arrays (Ps, qs, As, bs, Gs, lbs, ubs)
    """
    Ps, qs, As, bs, Gs, lbs, ubs = [], [], [], [], [], [], []

    for _ in range(n_batch):
        P, q, A, b, G, lb, ub = generate_qp_func(qp_size)

        # Convert sparse matrices to dense arrays
        Ps.append(P.toarray() if sp.issparse(P) else P)
        qs.append(q.toarray() if sp.issparse(q) else q)
        As.append(A.toarray() if sp.issparse(A) else A)
        bs.append(b.toarray() if sp.issparse(b) else b)
        Gs.append(G.toarray() if sp.issparse(G) else G)
        lbs.append(lb.toarray() if sp.issparse(lb) else lb)
        ubs.append(ub.toarray() if sp.issparse(ub) else ub)

    return (
        np.stack(Ps, axis=0),
        np.stack(qs, axis=0),
        np.stack(As, axis=0),
        np.stack(bs, axis=0),
        np.stack(Gs, axis=0),
        np.stack(lbs, axis=0),
        np.stack(ubs, axis=0),
    )


def solve_batch_qp_torch_feasible(
    P: torch.Tensor,
    q: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    G: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Solve a batch of QP problems using torch QPFunction with structural feasibility.

    Args:
        P: Batch of quadratic cost matrices
        q: Batch of linear cost vectors
        A: Batch of equality constraint matrices
        b: Batch of equality constraint vectors
        G: Batch of inequality constraint matrices
        lb: Batch of lower bounds for inequality constraints
        ub: Batch of upper bounds for inequality constraints

    Returns:
        Batch of optimal solution tensors
    """
    eps = 1e-5
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=True)
    output_torch = function(P, q, A, b, G, lb, ub)
    return output_torch[0]


def solve_batch_qp_torch_non_structural_feasible(
    P: torch.Tensor,
    q: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    G: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Solve a batch of QP problems using torch QPFunction without structural feasibility.

    Args:
        P: Batch of quadratic cost matrices
        q: Batch of linear cost vectors
        A: Batch of equality constraint matrices
        b: Batch of equality constraint vectors
        G: Batch of inequality constraint matrices
        lb: Batch of lower bounds for inequality constraints
        ub: Batch of upper bounds for inequality constraints

    Returns:
        Batch of optimal solution tensors
    """
    eps = 1e-5
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=False)
    output_torch = function(P, q, A, b, G, lb, ub)
    return output_torch[0]


def to_torch_tensors(
    arrays: Tuple[Union[np.ndarray, sp.spmatrix], ...],
) -> Tuple[torch.Tensor, ...]:
    """Convert arrays (dense or sparse) to torch tensors.

    Args:
        arrays: Tuple of numpy arrays or scipy sparse matrices

    Returns:
        Tuple of torch tensors (converted to dense if input was sparse)
    """
    tensors = []
    for arr in arrays:
        if sp.issparse(arr):
            arr = arr.toarray()
        tensors.append(torch.from_numpy(arr))
    return tuple(tensors)


def to_dense_np_arrays(
    arrays: Tuple[Union[np.ndarray, sp.spmatrix], ...],
) -> Tuple[np.ndarray, ...]:
    """Convert arrays (dense or sparse) to dense numpy arrays.

    Args:
        arrays: Tuple of numpy arrays or scipy sparse matrices

    Returns:
        Tuple of dense numpy arrays
    """
    dense_arrays = []
    for arr in arrays:
        if sp.issparse(arr):
            dense_arrays.append(arr.toarray())
        else:
            dense_arrays.append(arr)
    return tuple(dense_arrays)


def generate_mixed_qp(n: int, seed: int = 1, reg: float = 0.01) -> Tuple[
    sp.coo_matrix,
    np.ndarray,
    sp.csc_matrix,
    np.ndarray,
    sp.csc_matrix,
    np.ndarray,
    np.ndarray,
]:
    """Generate a mixed quadratic programming problem in QP format.

    Args:
        n: Problem dimension
        seed: Random seed for reproducibility
        reg: Regularization parameter to ensure positive definiteness

    Returns:
        Tuple containing (P, q, A_eq, b_eq, A_ineq, b_ineq, l_ineq)
        where the QP problem is:
            minimize    (1/2) x^T P x + q^T x
            subject to  A_eq x = b_eq
                       A_ineq x >= b_ineq
    """
    np.random.seed(seed)

    m = int(n / 4) + int(n / 4)
    n_eq = 1
    n_in = 1

    # Generate random positive semi-definite matrix P
    P = sp.random(n, n, density=0.075, data_rvs=np.random.randn, format="csc").toarray()
    P = (P + P.T) / 2.0

    # Ensure positive definiteness
    eigenvals = np.linalg.eigvals(P)
    s = max(np.abs(eigenvals))
    P += (abs(s) + reg) * sp.eye(n)
    P = sp.coo_matrix(P)

    # Generate linear term
    q = np.random.randn(n)

    # Generate constraint matrices
    A = sp.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    v = np.random.randn(n)
    delta = np.random.rand(m)
    u = A @ v
    l = 1.0e20 * np.ones(m)

    return P, q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


def is_qp_feasible(
    Q: np.ndarray,
    p: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    G: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> Optional[bool]:
    """Check if a quadratic programming problem is feasible.

    Args:
        Q: Quadratic cost matrix
        p: Linear cost vector
        A: Equality constraint matrix (optional)
        b: Equality constraint vector (optional)
        G: Inequality constraint matrix (optional)
        lb: Lower bounds for inequality constraints (optional)
        ub: Upper bounds for inequality constraints (optional)

    Returns:
        True if feasible, False if infeasible, None if solver status unclear
    """
    n = Q.shape[0]
    x = cp.Variable(n)

    constraints = []

    # Add equality constraints
    if A is not None and b is not None:
        constraints.append(A @ x == b)

    # Add inequality constraints
    if G is not None:
        if lb is not None:
            constraints.append(G @ x >= lb)
        if ub is not None:
            constraints.append(G @ x <= ub)

    # Create feasibility problem (minimize 0)
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return True
    elif prob.status == "infeasible":
        return False
    else:
        return None


class TestQpLayerWrapper(unittest.TestCase):
    """Test suite for QP layer functionality comparing different solvers."""

    def setUp(self) -> None:
        """Set up test fixtures with consistent random seed for reproducibility."""
        self.qp_size = 10
        self.batch_size = 10
        self.tolerance = 1e-5

    def test_single_qp_solver_consistency(self) -> None:
        """Test that different single QP solvers produce consistent results."""
        # Generate a single QP problem
        qp_matrices = generate_mixed_qp(self.qp_size)

        # Solve using different methods
        torch_sol_feasible = solve_single_qp_torch_feasible(
            *to_torch_tensors(qp_matrices)
        )
        torch_sol_non_structural_feasible = (
            solve_single_qp_torch_non_structural_feasible(
                *to_torch_tensors(qp_matrices)
            )
        )
        numpy_sol = solve_single_qp_numpy(*to_dense_np_arrays(qp_matrices))

        # Assert solutions match within tolerance
        self.assertTrue(
            np.allclose(
                torch_sol_feasible.detach().cpu().numpy(),
                numpy_sol,
                rtol=self.tolerance,
            ),
            "QPFunction does not match proxqp.dense.solve for single QP problem",
        )

        self.assertTrue(
            np.allclose(
                torch_sol_feasible.detach().cpu().numpy(),
                torch_sol_non_structural_feasible.detach().cpu().numpy(),
                rtol=self.tolerance,
            ),
            "Structural feasible and non-structural feasible solutions do not match",
        )

    def test_batch_qp_solver_consistency(self) -> None:
        """Test that batch QP solvers with different feasibility modes produce consistent results."""
        # Generate batch of QP problems
        batch = batch_generate_qps(generate_mixed_qp, self.batch_size, self.qp_size)

        # Solve using different feasibility modes
        sol_structural_feasible = solve_batch_qp_torch_feasible(
            *to_torch_tensors(batch)
        )
        sol_non_structural_feasible = solve_batch_qp_torch_non_structural_feasible(
            *to_torch_tensors(batch)
        )

        # Test batch solver against individual numpy solutions for validation
        batch_sol_torch = solve_batch_qp_torch_feasible(*to_torch_tensors(batch))

        # Solve each QP individually with numpy and concatenate results
        numpy_solutions = []
        for i in range(self.batch_size):
            single_qp = tuple(arr[i] for arr in batch)
            numpy_sol = solve_single_qp_numpy(*single_qp)
            numpy_solutions.append(numpy_sol)

        numpy_batch_sol = np.stack(numpy_solutions, axis=0)

        self.assertTrue(
            np.allclose(
                batch_sol_torch.detach().cpu().numpy(),
                numpy_batch_sol,
                rtol=self.tolerance,
            ),
            "Batch PyTorch solver does not match concatenated individual numpy solutions",
        )

        # Assert batch solutions match within tolerance
        self.assertTrue(
            np.allclose(
                sol_structural_feasible.detach().cpu().numpy(),
                sol_non_structural_feasible.detach().cpu().numpy(),
                rtol=self.tolerance,
            ),
            "Batch structural feasible and non-structural feasible solutions do not match",
        )

        # Verify batch dimension is correct
        expected_batch_shape = (self.batch_size, self.qp_size)
        self.assertEqual(
            sol_structural_feasible.shape,
            expected_batch_shape,
            f"Expected batch solution shape {expected_batch_shape}, got {sol_structural_feasible.shape}",
        )

    def test_backward_pass(self) -> None:
        """Test that backward pass works correctly for both single and batch problems."""

        # Test single QP backward pass with structural feasibility
        qp_matrices = generate_mixed_qp(self.qp_size)
        torch_tensors_single = to_torch_tensors(qp_matrices)

        # Enable gradients for parameters
        torch_tensors_single = tuple(
            t.requires_grad_(True) for t in torch_tensors_single
        )

        try:
            sol_single_feasible = solve_single_qp_torch_feasible(*torch_tensors_single)
            # Create a scalar loss to compute gradients
            loss = sol_single_feasible.sum()
            loss.backward()
        except Exception as e:
            self.fail(
                f"Single QP backward pass with structural feasibility failed: {e}"
            )

        # Test single QP backward pass without structural feasibility
        torch_tensors_single = to_torch_tensors(qp_matrices)
        torch_tensors_single = tuple(
            t.requires_grad_(True) for t in torch_tensors_single
        )

        try:
            sol_single_non_feasible = solve_single_qp_torch_non_structural_feasible(
                *torch_tensors_single
            )
            loss = sol_single_non_feasible.sum()
            loss.backward()
        except Exception as e:
            self.fail(
                f"Single QP backward pass without structural feasibility failed: {e}"
            )

        # Test batch QP backward pass with structural feasibility
        batch = batch_generate_qps(generate_mixed_qp, self.batch_size, self.qp_size)
        torch_tensors_batch = to_torch_tensors(batch)
        torch_tensors_batch = tuple(t.requires_grad_(True) for t in torch_tensors_batch)

        try:
            sol_batch_feasible = solve_batch_qp_torch_feasible(*torch_tensors_batch)
            loss = sol_batch_feasible.sum()
            loss.backward()
        except Exception as e:
            self.fail(f"Batch QP backward pass with structural feasibility failed: {e}")

        # Test batch QP backward pass without structural feasibility
        torch_tensors_batch = to_torch_tensors(batch)
        torch_tensors_batch = tuple(t.requires_grad_(True) for t in torch_tensors_batch)

        try:
            sol_batch_non_feasible = solve_batch_qp_torch_non_structural_feasible(
                *torch_tensors_batch
            )
            loss = sol_batch_non_feasible.sum()
            loss.backward()
        except Exception as e:
            self.fail(
                f"Batch QP backward pass without structural feasibility failed: {e}"
            )

        # If we reach here, all backward passes succeeded
        self.assertTrue(True, "All backward passes completed successfully")


if __name__ == "__main__":

    unittest.main()
