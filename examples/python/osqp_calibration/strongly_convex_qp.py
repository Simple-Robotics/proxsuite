# from calibration_base import test_calibration_qp

# # Run test
# test_calibration_qp(
#     problem="strongly_convex_qp",
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
# Failed: 2/50  | dim=10 iter 27 vs 30 | dim=30 30 vs 31

# only_in:
# Failed: 16/50 | dim=10 diff_x error 6e-2 and diff_yz error 2e-3 and iter 23 vs 19 |
#               | other dim: iter (max gap 2)

# n_eq and n_in:
# Failed: 3/50 | dim=10 iter 25 vs 28 | dim=250 34 vs 35 | dim=990 41 vs 42

# => Implem very close from source

# Polishing, only_eq / only_in:
# False, False: Pass
# True, False: Pass
# False, True: Same status and status_polish, but dim = 710 has polish => r_pri proxsuite = 1e-14 and source = 1e-5

# => In the big lines: Pass or better

# Mu updates:
# True, False:  OK (0 update, because solver solves before interval = 50).
#               Case interval = 10: Almost all good (only 2 dims where gap = 1 update)
# False, False: OK (0 update, because solver solves before interval = 50).
#               Case interval = 10: Almost all good (only 1 dim where gap = 1 update)
# False, True: OK (same results), except one case with diff = 1 mu update
#               Case interval = 10: Same, no problem

# => In the big lines: Pass
