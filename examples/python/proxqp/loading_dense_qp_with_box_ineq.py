import proxsuite

n = 10
n_eq = 2
n_in = 2
# we load here a QP model with
# n_eq equality constraints
# n_in generic type of inequality constraints
# and dim box inequality constraints
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in, True)
# true specifies we take into accounts box constraints
# n_in are any other type of inequality constraints

# Another example

# we load here a QP model with
# n_eq equality constraints
# O generic type of inequality constraints
# and dim box inequality constraints
qp2 = proxsuite.proxqp.dense.QP(n, n_eq, 0, True)
# true specifies we take into accounts box constraints
# we don't need to precise n_in = dim, it is taken
# into account internally
