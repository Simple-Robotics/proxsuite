from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="not_strongly_convex_qp",
    dim_start=10,
    dim_end=1000,
    dim_step=20,
    only_eq=False,
    only_in=False,
)

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
