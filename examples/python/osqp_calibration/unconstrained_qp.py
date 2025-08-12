from calibration_base import test_calibration_qp

# Run test
test_calibration_qp(
    problem="unconstrained_qp",
    dim_start=10,
    dim_end=1000,
    dim_step=20,
    only_eq=False,
    only_in=False,
)

# Notes:

# Failed: 0/50

# => Implem very close from source
