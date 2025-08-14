# from calibration_base import test_calibration_qp

# # Run test
# test_calibration_qp(
#     problem="not_strongly_convex_qp",
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
#     adaptive_mu_interval=50,
#     adaptive_mu_tolerance=5.0,
#     polishing=False,
#     delta_osqp=1e-6,
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
# Failed: 50/50 | iter error increases with dim, and max diff iter = 6 in favour of proxsuite

# only_in:
# Failed: 49/50 | iter error increases with dim with big diff in favour of proxsuite (eg 28 vs 208)
#               | dim=10, 30: diff_yz error 1e-3 | dim=50: diff_yz error 2e-3

# n_eq and n_in:
# Failed: 50/50 | iter error increases with dim with big diff in favour of proxsuite (eg 38 vs 183)
#               | dim=10: diff_yz error 1e-3

# => Errors in variable values are negligible
# => Errors in number of iterations suggest that proxsuite efficient and stable with increasing
# dim but not osqp source

# Note:
# We are in a not strong convexity setting regarding the hessian (H)
# # With ineq: Number of iters is stable (around 38 from some values of dim), but source incearses with dim
# # With eq only: Not this behaviour (and max 6 iter of difference)


# Polishing, only_eq / only_in:
# False, False: Pass
# True, False: Pass
# False, True: status polish different at dim = 390 and 430 => proxsuite sucess and source fails

# => In the big lines: Pass or better

# Mu updates:
# True, False: Same results, and no mu update (proxsuite and source)
# False, False: Still large number of iter fails | full mu updates fails from dim threshold (proxsuite dont and source does)
# False, True: Same
# Interpretation: Comes from the fact that proxsuite is stopped way before ?
