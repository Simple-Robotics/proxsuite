from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="degenerate_qp",
    dim_start=10,
    dim_end=1000,
    dim_step=20,
    only_eq=False,
    only_in=False,
    max_iter=4000,
    compute_preconditioner=True,
    eps_abs=1e-3,
    eps_rel=0,
    sparsity_factor=0.45,
    strong_convexity_factor=1e-2,
    adaptive_mu=False,
    polishing=False,
    verbose_solver=False,
    verbose_results_variables=False,
    verbose_calibration=False,
    verbose_timings=False,
    prec_x=1e-3,
    prec_yz=1e-3,
    prec_r_pri=1e-3,
    prec_r_dua=1e-3,
    prec_iter=0,
)

# Notes:

# only_eq:
# Failed: 2/50  | We retieve case of strongly convex qp, only few differences in number iter

# only_in:
# Failed: 47/50 | status: Primal infeasible vs solved (36/50)
#               | iter: proxsuite stops (way) before source (47/50)

# n_eq: and n_in
# Failed: /50   | Similar to only_in

# => only_eq: Trivial and out of discussion
# => proxsuite detects primal infeasibility and stops early, while source can go up to 3000 iter to solve

# Case where I early stop (eg after 20 iter):
# Proxsuite residuals > (>>) to source residual.
# With dim increasing: This difference (ratio) vanishes
# Intuition ?
