import proxsuite
import numpy as np
from util import generate_mixed_qp


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
H, g, A, b, C, u, l = generate_mixed_qp(n, True)

# solve the problem using the sparse backend
results = proxsuite.proxqp.sparse.solve(H, g, A, b, C, l, u)

# solve the problem using the dense backend

results2 = proxsuite.proxqp.dense.solve(
    H.toarray(order="C"), g, A.toarray(order="C"), b, C.toarray(order="C"), l, u
)
# Note finally, that the matrices are in sparse format, when using the dense backend you
# should convert them in dense format

# print an optimal solution
print("optimal x: {}".format(results.x))
print("optimal y: {}".format(results.y))
print("optimal z: {}".format(results.z))

# solve the problem using the dense backend using its feature for handling box constraints

l_box = -np.ones(n) * 1.0e10
u_box = np.ones(n) * 1.0e10

# make sure to specify l_box=l_box, u_box=u_box in order to make work the
# overloading
results_dense_solver_box = proxsuite.proxqp.dense.solve(
    H.toarray(order="C"),
    g,
    A.toarray(order="C"),
    b,
    C.toarray(order="C"),
    l,
    u,
    l_box=l_box,
    u_box=u_box,
)
# print an optimal solution
print("optimal x: {}".format(results_dense_solver_box.x))
print("optimal y: {}".format(results_dense_solver_box.y))
print("optimal z: {}".format(results_dense_solver_box.z))
# Note that the last n elements of z corresponds to the multipliers associated to the box
# constraints
