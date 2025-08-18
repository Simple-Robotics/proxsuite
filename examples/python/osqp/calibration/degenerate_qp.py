# from calibration_base import test_calibration_qp

# # Run test
# test_calibration_qp(
#     problem="degenerate_qp",
#     dim_start=10,
#     dim_end=1000,
#     dim_step=20,
#     only_eq=False,
#     only_in=False,
#     max_iter=4000,
#     compute_preconditioner=True,
#     eps_abs=1e-3,
#     eps_rel=0,
#     eps_primal_inf=1e-15,
#     eps_dual_inf=1e-15,
#     sparsity_factor=0.45,
#     strong_convexity_factor=1e-2,
#     adaptive_mu=False,
#     adaptive_mu_interval=50,
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
# Failed: 2/50  | We retieve case of strongly convex qp, only few differences in number iter

# only_in:
# Failed: 47/50 | status: Primal infeasible vs solved (36/50)
#               | iter: proxsuite stops (way) before source (47/50)

# n_eq: and n_in
# Failed: 50/50 | Similar to only_in

# => only_eq: Trivial and out of discussion
# => proxsuite detects primal infeasibility and stops early, while source can go up to 3000 iter to solve

# Case where I early stop (eg after 20 iter):
# Proxsuite residuals > (>>) to source residual.
# With dim increasing: This difference (ratio) vanishes
# Intuition ?

# Note: With eps_pri_inf = 1e-12, eps_dua_inf = 1e-12:
# False, False: Light diff on iter (max 9)
# False, True: Similar

# mu updates on False, False:
# without mu updates: dim = 590 on diff yz (error 3e-3) and diff iter (9), and others in iter
# diff yz in [190, 410, 450, 590, 670, 770, 890, 990]
# diff mu updates in [190, 330, 830, 870]
# dim = 190: All pass without update,
#            but fails with: diff yz max error = 0.17, and iter 203 vs 91, and mu updates 3 vs 1
# dim = 330: Without: iter 1863 vs 1856 | With: iter 233 vs 206, mu updates 2 vs 3
# dim = 830: Without: iter 811 vs 810 | With: iter 162 vs 198, mu updates 2 vs 1
# dim = 870: Without: iter 682 vs 679 | With: iter 202 vs 191, mu updates 2 vs 1
# others dims that dont fail: 46 over 50 test cases give the same number of updates
#                             42 over 50 give same yz, error max 1e-1, a lot env 1e-2 or 1e-3
#                  with polishing on top of this: 46 tests pass (except iter)

# => In the big lines: Pass
