from typing import Optional, Tuple, Union
import unittest

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
    eq: bool = True,
    neq: bool = True,
    feasible=True,
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

    """
    if not eq:
        A = b = None
    if not neq:
        C = u = l = None
    qp = proxsuite.proxqp.dense.QP(
        H.shape[0],
        A.shape[0] if not A is None else 0,
        C.shape[0] if not C is None else 0,
    )
    qp.settings.eps_abs = 1e-9
    qp.settings.max_iter = 1000
    qp.settings.max_iter_in = 1000
    default_rho = 5.0e-5
    qp.settings.default_rho = default_rho
    qp.settings.refactor_rho_threshold = default_rho

    qp.init(
        H=H,
        g=np.asfortranarray(g),
        A=A,
        b=np.asfortranarray(b) if b is not None else None,
        C=C,
        l=np.asfortranarray(l) if l is not None else None,
        u=np.asfortranarray(u) if u is not None else None,
        rho=default_rho,
    )
    qp.settings.primal_infeasibility_solving = not feasible
    qp.solve()
    return qp.results.x


def solve_single_qp_derivative_numpy(
    H: np.ndarray,
    g: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    C: Optional[np.ndarray],
    l: Optional[np.ndarray],
    u: Optional[np.ndarray],
    eq: bool = True,
    neq: bool = True,
    feasible=True,
) -> np.ndarray:
    """Solve a single QP problem using proxsuite numpy backend then compute the gradients of the QP matrices with respect to the squared norm of the QP output.

    Args:
        H: Quadratic cost matrix (Hessian)
        g: Linear cost vector
        A: Equality constraint matrix
        b: Equality constraint vector
        C: Inequality constraint matrix
        l: Lower bounds for inequality constraints
        u: Upper bounds for inequality constraints

    Returns:
        gradients on QP matrices

    """
    if not eq:
        A = b = None
    if not neq:
        C = u = l = None
    qp = proxsuite.proxqp.dense.QP(
        H.shape[0],
        A.shape[0] if not A is None else 0,
        C.shape[0] if not C is None else 0,
    )
    qp.settings.eps_abs = 1e-9
    qp.settings.max_iter = 1000
    qp.settings.max_iter_in = 1000
    default_rho = 5.0e-5
    qp.settings.default_rho = default_rho
    qp.settings.refactor_rho_threshold = default_rho
    qp.init(
        H=H,
        g=np.asfortranarray(g),
        A=A,
        b=np.asfortranarray(b) if b is not None else None,
        C=C,
        l=np.asfortranarray(l) if l is not None else None,
        u=np.asfortranarray(u) if u is not None else None,
        rho=default_rho,
    )
    qp.settings.primal_infeasibility_solving = not feasible
    qp.solve()
    output_gradient = np.zeros(
        H.shape[0]
        + (A.shape[0] if A is not None else 0)
        + (C.shape[0] if C is not None else 0)
    )

    output_gradient[: H.shape[0]] = 2 * qp.results.x
    proxsuite.proxqp.dense.compute_backward(qp, output_gradient, 1e-4, 1e-6, 1e-6)
    return (
        qp.model.backward_data.dL_dH,
        qp.model.backward_data.dL_dg,
        qp.model.backward_data.dL_dA,
        qp.model.backward_data.dL_db,
        qp.model.backward_data.dL_dC,
        qp.model.backward_data.dL_dl,
        qp.model.backward_data.dL_du,
    )


def solve_single_qp_torch_feasible(
    P: torch.Tensor,
    q: torch.Tensor,
    A: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    G: Optional[torch.Tensor],
    lb: Optional[torch.Tensor],
    ub: Optional[torch.Tensor],
    eq=True,
    neq=True,
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
    eps = 1e-9
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=True)
    if not eq:
        A = b = torch.tensor([])
    if not neq:
        G = lb = ub = torch.tensor([])
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
    eq=True,
    neq=True,
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
    eps = 1e-9
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=False)
    if not eq:
        A = b = torch.tensor([])
    if not neq:
        G = ub = lb = torch.tensor([])
    output_torch = function(P, q, A, b, G, lb, ub)
    return output_torch[0]


def batch_generate_qps(
    generate_qp_func: callable, n_batch: int, qp_size: int = 10, seed=1
) -> Tuple[np.ndarray, ...]:
    """Generate a batch of QP problems for testing.

    Args:
        generate_qp_func: Function to generate a single QP problem
        n_batch: Number of QP problems to generate
        qp_size: Size of each QP problem
        seed: Random seed for generate_qp_func

    Returns:
        Tuple of stacked numpy arrays (Ps, qs, As, bs, Gs, lbs, ubs)
    """
    Ps, qs, As, bs, Gs, lbs, ubs = [], [], [], [], [], [], []

    for _ in range(n_batch):
        P, q, A, b, G, lb, ub = generate_qp_func(qp_size, seed)
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
    eq=True,
    neq=True,
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
    eps = 1e-9
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=True)
    if not eq:
        A = b = torch.tensor([])
    if not neq:
        G = lb = ub = torch.tensor([])
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
    eq=True,
    neq=True,
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
    eps = 1e-9
    max_iter = 1000
    function = qplayer.QPFunction(eps, max_iter, structural_feasibility=False)
    if not eq:
        A = b = torch.tensor([])
    if not neq:
        G = lb = ub = torch.tensor([])
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
    u = A @ v
    l = 1.0e20 * np.ones(m)

    return P, q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


class TestQpLayerWrapper(unittest.TestCase):
    """Test suite for QP layer functionality comparing different solvers."""

    def setUp(self) -> None:
        """Set up test fixtures with consistent random seed for reproducibility."""
        self.qp_size = 10
        self.batch_size = 10
        self.tolerance = 1e-9
        self.grad_tolerance = 1e-5

    def test_single_qp_solver_consistency(self) -> None:
        for seed in range(10):
            """Test that different single QP solvers produce consistent results."""
            # Generate a single QP problem

            qp_matrices = generate_mixed_qp(self.qp_size, seed=seed)

            # Solve using different methods
            # With eq and neq
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
                    atol=self.tolerance,
                ),
                "QPFunction does not match proxqp.dense.solve for single QP problem with eq and neq",
            )
            self.assertTrue(
                np.allclose(
                    torch_sol_feasible.detach().cpu().numpy(),
                    torch_sol_non_structural_feasible.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Structural feasible and non-structural feasible solutions do not match with neq and eq",
            )

            # With eq
            numpy_sol = solve_single_qp_numpy(
                *to_dense_np_arrays(qp_matrices), eq=True, neq=False
            )

            torch_sol_feasible = solve_single_qp_torch_feasible(
                *to_torch_tensors(qp_matrices), eq=True, neq=False
            )
            torch_sol_non_structural_feasible = (
                solve_single_qp_torch_non_structural_feasible(
                    *to_torch_tensors(qp_matrices), eq=True, neq=False
                )
            )

            self.assertTrue(
                np.allclose(
                    torch_sol_feasible.detach().cpu().numpy(),
                    numpy_sol,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "QPFunction does not match proxqp.dense.solve for single QP problem with eq and no neq",
            )

            self.assertTrue(
                np.allclose(
                    torch_sol_feasible.detach().cpu().numpy(),
                    torch_sol_non_structural_feasible.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Structural feasible and non-structural feasible solutions do not match with neq and eq",
            )

            # With neq
            numpy_sol = solve_single_qp_numpy(
                *to_dense_np_arrays(qp_matrices), eq=False, neq=True
            )
            torch_sol_feasible = solve_single_qp_torch_feasible(
                *to_torch_tensors(qp_matrices), eq=False, neq=True
            )

            torch_sol_non_structural_feasible = (
                solve_single_qp_torch_non_structural_feasible(
                    *to_torch_tensors(qp_matrices), eq=False, neq=True
                )
            )

            self.assertTrue(
                np.allclose(
                    torch_sol_feasible.detach().cpu().numpy(),
                    numpy_sol,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "QPFunction does not match proxqp.dense.solve for single QP problem with neq and no eq",
            )

            self.assertTrue(
                np.allclose(
                    torch_sol_feasible.detach().cpu().numpy(),
                    torch_sol_non_structural_feasible.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Structural feasible and non-structural feasible solutions do not match with neq and no eq",
            )

            # TODO
            # Without eq or neq
            # I am not sure that without eq or neq is an intended mode as the original code has :
            #                           "assert neq > 0 or nineq > 0"
            # for now we consider that this mode is not doable

    def test_batch_qp_solver_consistency(self) -> None:
        for seed in range(10):
            """Test that batch QP solvers with different feasibility modes produce consistent results."""
            # Generate batch of QP problems
            batch = batch_generate_qps(
                generate_mixed_qp, self.batch_size, self.qp_size, seed=seed
            )

            # Solve using different feasibility modes
            # With eq and neq
            sol_structural_feasible = solve_batch_qp_torch_feasible(
                *to_torch_tensors(batch)
            )
            sol_non_structural_feasible = solve_batch_qp_torch_non_structural_feasible(
                *to_torch_tensors(batch)
            )

            # Test batch solver against individual numpy solutions for validation

            # Solve each QP individually with numpy and concatenate results
            numpy_solutions = []
            for i in range(self.batch_size):
                single_qp = tuple(arr[i] for arr in batch)
                numpy_sol = solve_single_qp_numpy(*single_qp)
                numpy_solutions.append(numpy_sol)

            numpy_batch_sol = np.stack(numpy_solutions, axis=0)

            self.assertTrue(
                np.allclose(
                    sol_structural_feasible.detach().cpu().numpy(),
                    numpy_batch_sol,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Batch PyTorch solver does not match concatenated individual numpy solutions",
            )
            # Assert batch solutions match within tolerance
            self.assertTrue(
                np.allclose(
                    sol_structural_feasible.detach().cpu().numpy(),
                    sol_non_structural_feasible.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
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

            # With eq
            sol_structural_feasible = solve_batch_qp_torch_feasible(
                *to_torch_tensors(batch), eq=True, neq=False
            )
            sol_non_structural_feasible = solve_batch_qp_torch_non_structural_feasible(
                *to_torch_tensors(batch), eq=True, neq=False
            )

            # Test batch solver against individual numpy solutions for validation
            batch_sol_torch = solve_batch_qp_torch_feasible(
                *to_torch_tensors(batch), eq=True, neq=False
            )

            # Solve each QP individually with numpy and concatenate results
            numpy_solutions = []
            for i in range(self.batch_size):
                single_qp = tuple(arr[i] for arr in batch)
                numpy_sol = solve_single_qp_numpy(*single_qp, eq=True, neq=False)
                numpy_solutions.append(numpy_sol)

            numpy_batch_sol = np.stack(numpy_solutions, axis=0)

            self.assertTrue(
                np.allclose(
                    batch_sol_torch.detach().cpu().numpy(),
                    numpy_batch_sol,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Batch PyTorch solver does not match concatenated individual numpy solutions with eq and no neq",
            )

            # Assert batch solutions match within tolerance
            self.assertTrue(
                np.allclose(
                    sol_structural_feasible.detach().cpu().numpy(),
                    sol_non_structural_feasible.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Batch structural feasible and non-structural feasible solutions do not match with eq and no neq",
            )

            # With neq
            sol_structural_feasible = solve_batch_qp_torch_feasible(
                *to_torch_tensors(batch), eq=False, neq=True
            )
            sol_non_structural_feasible = solve_batch_qp_torch_non_structural_feasible(
                *to_torch_tensors(batch), eq=False, neq=True
            )

            # Test batch solver against individual numpy solutions for validation
            batch_sol_torch = solve_batch_qp_torch_feasible(
                *to_torch_tensors(batch), eq=False, neq=True
            )

            # Solve each QP individually with numpy and concatenate results
            numpy_solutions = []
            for i in range(self.batch_size):
                single_qp = tuple(arr[i] for arr in batch)
                numpy_sol = solve_single_qp_numpy(*single_qp, eq=False, neq=True)
                numpy_solutions.append(numpy_sol)

            numpy_batch_sol = np.stack(numpy_solutions, axis=0)

            self.assertTrue(
                np.allclose(
                    batch_sol_torch.detach().cpu().numpy(),
                    numpy_batch_sol,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Batch PyTorch solver does not match concatenated individual numpy solutions with neq and no eq",
            )

            # Assert batch solutions match within tolerance
            self.assertTrue(
                np.allclose(
                    sol_structural_feasible.detach().cpu().numpy(),
                    sol_non_structural_feasible.detach().cpu().numpy(),
                    rtol=self.tolerance,
                    atol=self.tolerance,
                ),
                "Batch structural feasible and non-structural feasible solutions do not match with neq and no eq",
            )

    def test_backward_pass(self) -> None:
        for seed in range(10):
            """Test that backward pass works correctly for both single and batch problems."""

            # With eq and neq

            # Test single QP backward pass with structural feasibility
            qp_matrices = generate_mixed_qp(self.qp_size, seed=seed)
            torch_tensors_single = to_torch_tensors(qp_matrices)

            # Enable gradients for parameters
            torch_tensors_single = tuple(
                t.requires_grad_(True) for t in torch_tensors_single
            )

            try:
                sol_single_feasible = solve_single_qp_torch_feasible(
                    *torch_tensors_single
                )
                # Create a scalar loss to compute gradients
                loss = sol_single_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices)
                )
                for i, tensor in enumerate(torch_tensors_single):
                    self.assertTrue(
                        np.allclose(
                            tensor.grad.detach().cpu().numpy(),
                            grad_np[i],
                            atol=self.grad_tolerance,
                            rtol=self.grad_tolerance,
                        )
                    )
            except Exception as e:
                self.fail(
                    f"Single QP backward pass with structural feasibility failed (with eq and neq): {e}"
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
                loss = sol_single_non_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), feasible=False
                )
                for i, tensor in enumerate(torch_tensors_single):
                    if i < 0:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad.detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )
            except Exception as e:
                self.fail(
                    f"Single QP backward pass without structural feasibility failed (with eq and neq): {e}"
                )

            # Test batch QP backward pass with structural feasibility
            batch = batch_generate_qps(
                generate_mixed_qp, self.batch_size, self.qp_size, seed=seed
            )
            qp_matrices = [matrix[0] for matrix in batch]
            torch_tensors_batch = to_torch_tensors(batch)
            torch_tensors_batch = tuple(
                t.requires_grad_(True) for t in torch_tensors_batch
            )

            try:
                sol_batch_feasible = solve_batch_qp_torch_feasible(*torch_tensors_batch)
                loss = sol_batch_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices)
                )
                for i, tensor in enumerate(torch_tensors_batch):
                    if i < 0:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad[0].detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )

            except Exception as e:
                self.fail(
                    f"Batch QP backward pass with structural feasibility failed (with eq and neq): {e}"
                )

            # Test batch QP backward pass without structural feasibility
            torch_tensors_batch = to_torch_tensors(batch)
            torch_tensors_batch = tuple(
                t.requires_grad_(True) for t in torch_tensors_batch
            )

            try:
                sol_batch_non_feasible = solve_batch_qp_torch_non_structural_feasible(
                    *torch_tensors_batch
                )
                loss = sol_batch_non_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), feasible=False
                )
                for i, tensor in enumerate(torch_tensors_batch):
                    if i < 0:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad[0].detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )
            except Exception as e:
                self.fail(
                    f"Batch QP backward pass without structural feasibility failed (with eq and neq): {e}"
                )

            # With eq

            # Test single QP backward pass with structural feasibility
            qp_matrices = generate_mixed_qp(self.qp_size, seed=seed)
            torch_tensors_single = to_torch_tensors(qp_matrices)

            # Enable gradients for parameters
            torch_tensors_single = tuple(
                t.requires_grad_(True) for t in torch_tensors_single
            )

            try:
                sol_single_feasible = solve_single_qp_torch_feasible(
                    *torch_tensors_single, eq=True, neq=False
                )
                # Create a scalar loss to compute gradients
                loss = sol_single_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=True, neq=False
                )
                for i, tensor in enumerate(torch_tensors_single):
                    if tensor.grad is not None:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad.detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )
            except Exception as e:
                self.fail(
                    f"Single QP backward pass with structural feasibility failed (with eq and no neq): {e}"
                )

            # Test single QP backward pass without structural feasibility
            torch_tensors_single = to_torch_tensors(qp_matrices)
            torch_tensors_single = tuple(
                t.requires_grad_(True) for t in torch_tensors_single
            )

            try:
                sol_single_non_feasible = solve_single_qp_torch_non_structural_feasible(
                    *torch_tensors_single, eq=True, neq=False
                )
                loss = sol_single_non_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=True, neq=False, feasible=False
                )
                for i, tensor in enumerate(torch_tensors_single):
                    if i < 0:
                        if tensor.grad is not None:
                            self.assertTrue(
                                np.allclose(
                                    tensor.grad.detach().cpu().numpy(),
                                    grad_np[i],
                                    atol=self.grad_tolerance,
                                    rtol=self.grad_tolerance,
                                )
                            )
            except Exception as e:
                self.fail(
                    f"Single QP backward pass without structural feasibility failed (with eq and no neq): {e}"
                )

            # Test batch QP backward pass with structural feasibility
            batch = batch_generate_qps(
                generate_mixed_qp, self.batch_size, self.qp_size, seed=seed
            )
            qp_matrices = [matrix[0] for matrix in batch]
            torch_tensors_batch = to_torch_tensors(batch)
            torch_tensors_batch = tuple(
                t.requires_grad_(True) for t in torch_tensors_batch
            )

            try:
                sol_batch_feasible = solve_batch_qp_torch_feasible(
                    *torch_tensors_batch, eq=True, neq=False
                )
                loss = sol_batch_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=True, neq=False
                )
                for i, tensor in enumerate(torch_tensors_batch):
                    if i < 0 and tensor.grad is not None:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad[0].detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )

            except Exception as e:
                self.fail(
                    f"Batch QP backward pass with structural feasibility failed (with eq and no neq): {e}"
                )

            # Test batch QP backward pass without structural feasibility
            torch_tensors_batch = to_torch_tensors(batch)
            torch_tensors_batch = tuple(
                t.requires_grad_(True) for t in torch_tensors_batch
            )

            try:
                sol_batch_non_feasible = solve_batch_qp_torch_non_structural_feasible(
                    *torch_tensors_batch, eq=True, neq=False
                )
                loss = sol_batch_non_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=True, neq=False, feasible=False
                )
                for i, tensor in enumerate(torch_tensors_batch):
                    if i < 0 and tensor.grad is not None:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad[0].detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )
            except Exception as e:
                self.fail(
                    f"Batch QP backward pass without structural feasibility failed (with eq and no neq): {e}"
                )

            # With neq

            # Test single QP backward pass with structural feasibility
            qp_matrices = generate_mixed_qp(self.qp_size, seed=seed)
            torch_tensors_single = to_torch_tensors(qp_matrices)

            # Enable gradients for parameters
            torch_tensors_single = tuple(
                t.requires_grad_(True) for t in torch_tensors_single
            )

            try:
                sol_single_feasible = solve_single_qp_torch_feasible(
                    *torch_tensors_single, eq=False, neq=True
                )
                # Create a scalar loss to compute gradients
                loss = sol_single_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=False, neq=True
                )
                for i, tensor in enumerate(torch_tensors_single):
                    if tensor.grad is not None:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad.detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )
            except Exception as e:
                self.fail(
                    f"Single QP backward pass with structural feasibility failed (with neq and no eq): {e}"
                )

            # Test single QP backward pass without structural feasibility
            torch_tensors_single = to_torch_tensors(qp_matrices)
            torch_tensors_single = tuple(
                t.requires_grad_(True) for t in torch_tensors_single
            )

            try:
                sol_single_non_feasible = solve_single_qp_torch_non_structural_feasible(
                    *torch_tensors_single, eq=False, neq=True
                )
                loss = sol_single_non_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=False, neq=True, feasible=False
                )
                for i, tensor in enumerate(torch_tensors_single):
                    if i < 0:
                        if tensor.grad is not None:
                            self.assertTrue(
                                np.allclose(
                                    tensor.grad.detach().cpu().numpy(),
                                    grad_np[i],
                                    atol=self.grad_tolerance,
                                    rtol=self.grad_tolerance,
                                )
                            )
            except Exception as e:
                self.fail(
                    f"Single QP backward pass without structural feasibility failed (with neq and no eq): {e}"
                )

            # Test batch QP backward pass with structural feasibility
            batch = batch_generate_qps(
                generate_mixed_qp, self.batch_size, self.qp_size, seed=seed
            )
            qp_matrices = [matrix[0] for matrix in batch]
            torch_tensors_batch = to_torch_tensors(batch)
            torch_tensors_batch = tuple(
                t.requires_grad_(True) for t in torch_tensors_batch
            )

            try:
                sol_batch_feasible = solve_batch_qp_torch_feasible(
                    *torch_tensors_batch, eq=False, neq=True
                )
                loss = sol_batch_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=False, neq=True
                )
                for i, tensor in enumerate(torch_tensors_batch):
                    if i < 0 and tensor.grad is not None:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad[0].detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )

            except Exception as e:
                self.fail(
                    f"Batch QP backward pass with structural feasibility failed (with neq and no eq): {e}"
                )

            # Test batch QP backward pass without structural feasibility
            torch_tensors_batch = to_torch_tensors(batch)
            torch_tensors_batch = tuple(
                t.requires_grad_(True) for t in torch_tensors_batch
            )

            try:
                sol_batch_non_feasible = solve_batch_qp_torch_non_structural_feasible(
                    *torch_tensors_batch, eq=False, neq=True
                )
                loss = sol_batch_non_feasible.square().sum()
                loss.backward()
                grad_np = solve_single_qp_derivative_numpy(
                    *to_dense_np_arrays(qp_matrices), eq=False, neq=True, feasible=False
                )
                for i, tensor in enumerate(torch_tensors_batch):
                    if i < 0 and tensor.grad is not None:
                        self.assertTrue(
                            np.allclose(
                                tensor.grad[0].detach().cpu().numpy(),
                                grad_np[i],
                                atol=self.grad_tolerance,
                                rtol=self.grad_tolerance,
                            )
                        )
            except Exception as e:
                self.fail(
                    f"Batch QP backward pass without structural feasibility failed (with neq and no eq): {e}"
                )

        # If we reach here, all backward passes succeeded
        self.assertTrue(True, "All backward passes completed successfully")


if __name__ == "__main__":
    unittest.main()
