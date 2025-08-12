from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="primal_infeasible_qp",
    dim_start=10,
    dim_end=1000,
    dim_step=20,
    only_eq=False,
    only_in=True,
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
# Not tested because the infeasibility is build with inequality constraints

# only_in:
# dim = 90 fails with 20 vs 58 iter infavour of proxsuite, all primal infeasible
# dim = 10 fails with 39 vs 17 iter infavour of source, all primal infeasible

# n_eq and n_in:
# All tests pass

# => At prec_iter = 1, almost al tests pass on status + number of iter
# => All tests pass with iprec_iter = 0 on status
