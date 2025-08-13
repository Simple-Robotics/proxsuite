from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="unconstrained_qp",
    dim_start=10,
    dim_end=1000,
    dim_step=20,
    max_iter=4000,
    compute_preconditioner=True,
    eps_abs=1e-3,
    eps_rel=0,
    eps_primal_inf=1e-4,
    eps_dual_inf=1e-4,
    sparsity_factor=0.45,
    strong_convexity_factor=1e-2,
    adaptive_mu=False,
    polishing=False,
    delta_osqp=1e-6,
    polish_refine_iter=3,
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
)

# Notes:

# Failed: 0/50

# => Implem very close from source

# Polish:
# prec_polish = 1e-9
# => Pass
