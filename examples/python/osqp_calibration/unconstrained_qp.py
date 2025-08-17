# from calibration_base import test_calibration_qp

# # Run test
# test_calibration_qp(
#     problem="unconstrained_qp",
#     dim_start=10,
#     dim_end=1000,
#     dim_step=20,
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

# Failed: 0/50

# => Implem very close from source

# Polishing
# => Pass

# Mu updates:
# Case unconstrained, errors only on mu_updates
# Proxsuite (all): mu updates (1)
# Source (all): no mu updates (0)

# Rq: At each case, proxsuite updates at very first iter to maximum value of mu

# max_iter = 4
# [iteration 1]
# | primal residual=0.00e+00 | dual residual=3.19e+00 | duality gap=0.00e+00 | mu_eq=1.00e-02 | mu_in=1.00e+01
# [iteration 2]
# | primal residual=0.00e+00 | dual residual=1.91e+00 | duality gap=2.83e+02 | mu_eq=1.00e+03 | mu_in=1.00e+06
# [iteration 3]
# | primal residual=0.00e+00 | dual residual=1.15e+00 | duality gap=-6.81e+01 | mu_eq=1.00e+03 | mu_in=1.00e+06
# [iteration 4]
# | primal residual=0.00e+00 | dual residual=6.89e-01 | duality gap=7.70e+01 | mu_eq=1.00e+03 | mu_in=1.00e+06
# iter   objective    pri res    dua res    rho
#    1  -9.5226e+01   0.00e+00   1.91e+00   1.00e-01
#    2  -1.2929e+02   0.00e+00   1.15e+00   1.00e-01
#    3  -1.4148e+02   0.00e+00   6.89e-01   1.00e-01
#    4  -1.4584e+02   0.00e+00   4.14e-01   1.00e-01

# Note: Given the first change, the following since to be very close
# Yet what would come in the next update ? Different value and iter

# => In the big lines, same results (except diff of one update at the beginning)

# Points of interest to fix this:
# Computation of residuals
