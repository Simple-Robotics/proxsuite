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
    eps_primal_inf=1e-4,
    eps_dual_inf=1e-4,
    sparsity_factor=0.45,
    strong_convexity_factor=1e-2,
    adaptive_mu=False,
    adaptive_mu_interval=50,
    adaptive_mu_tolerance=5.0,
    polishing=False,
    delta_osqp=1e-6,
    polish_refine_iter=3,
    verbose_test_settings=True,
    verbose_solver=False,
    verbose_results_variables=False,
    verbose_calibration=False,
    verbose_timings=False,
    prec_x=1e-3,
    prec_yz=1e-3,
    prec_r_pri=1e-3,
    prec_r_dua=1e-3,
    prec_polish=1e-9,
    prec_iter=0,
    prec_mu_updates=0,
)

# Notes:

# => Implem very close from source
