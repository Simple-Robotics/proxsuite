# from calibration_base import test_calibration_qp

# # Run test
# test_calibration_qp(
#     problem="primal_infeasible_qp",
#     dim_start=10,
#     dim_end=1000,
#     dim_step=20,
#     only_eq=False,
#     only_in=False,
#     max_iter=4000,
#     compute_preconditioner=True,
#     eps_abs=1e-3,
#     eps_rel=0,
#     eps_primal_inf=1e-4,
#     eps_dual_inf=1e-4,
#     sparsity_factor=0.45,
#     strong_convexity_factor=1e-2,
#     adaptive_mu=False,
#     adaptive_mu_interval=10,
#     adaptive_mu_tolerance=5.0,
#     polish=False,
#     delta=1e-6,
#     polish_refine_iter=3,
#     verbose_test_settings=True,
#     verbose_solver=False,
#     verbose_results_variables=False,
#     verbose_calibration=False,
#     verbose_timings=False,
#     prec_x=1e-3,
#     prec_yz=1e-3,
#     prec_r_pri=1e-3,
#     prec_r_dua=1e-3,
#     prec_polish=1e-9,
#     prec_iter=0,
#     prec_mu_updates=0,
# )

# Notes:

# only_eq:
# All tests pass (at one iter)

# only_in:
# dim = 90 fails with 20 vs 58 iter infavour of proxsuite, all primal infeasible
# dim = 10 fails with 39 vs 17 iter infavour of source, all primal infeasible

# n_eq and n_in:
# All tests pass (at one iter)

# => At prec_iter = 1, almost al tests pass on status + number of iter
# => All tests pass with prec_iter = 0 on status

# Mu updates:
# True, False: Trivial (iter <50)
#              Case interval = 10: 2 errors over 50
# False, False: Case interval = 10: Diff seulement sur iter (1) et mu_updates (1), mais env 20 fails
# False, True: Case interval = 10: Much more errors iter (1) and just one eror mu update (1)

# => In the big lines: Quite OK
