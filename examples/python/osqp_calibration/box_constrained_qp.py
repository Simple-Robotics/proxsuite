from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="box_constrained_qp",
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

# only_eq:
# Not tested (makes no sense)

# only_in:
# Failed: 47/50 | dim=10 diff x 5e-3, diff y 0.8 (one coord), diff r_pri 7e-3,
#                 diff iter 205 vs 376, primal inf vs solved
#               | some dims with iter gap = 1 only, other with way larger
#               | proxsuite always better in terms of iter

# n_eq and n_in:
# Same that only_in

# Note:
# We are in a non strong convexity setting regarding the constraints (C)
