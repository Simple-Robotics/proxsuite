import proxsuite

n = 10
n_eq = 2
n_in = 2
# we load here a QP model with
# n_eq equality constraints
# n_in generic type of inequality constraints
# and dim box inequality constraints
#  we specify PrimalDualLdl backend
qp = proxsuite.proxqp.dense.QP(
    n, n_eq, n_in, True, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLdl
)
# true specifies we take into accounts box constraints
# n_in are any other type of inequality constraints

# Other examples

# we load here a QP model with
# n_eq equality constraints
# O generic type of inequality constraints
# and dim box inequality constraints
#  we specify PrimalLdl backend
qp2 = proxsuite.proxqp.dense.QP(
    n, n_eq, 0, True, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalLdl
)
# true specifies we take into accounts box constraints
# we don't need to precise n_in = dim, it is taken
# into account internally
# We let finally the solver decide
qp3 = proxsuite.proxqp.dense.QP(
    n, n_eq, 0, True, dense_backend=proxsuite.proxqp.dense.DenseBackend.Automatic
)
# Note that it is the default choice
