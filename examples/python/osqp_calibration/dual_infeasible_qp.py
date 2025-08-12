from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="dual_infeasible_qp",
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
    verbose_calibration=True,
    verbose_timings=False,
    prec_x=1e-3,
    prec_yz=1e-3,
    prec_r_pri=1e-3,
    prec_r_dua=1e-3,
    prec_iter=1,
)

# Notes:

# only_eq:
#

# only_in:
#

# n_eq and n_in:
#
