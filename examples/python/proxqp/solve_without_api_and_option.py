import proxsuite
from util import generate_mixed_qp


# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
H, g, A, b, C, u, l = generate_mixed_qp(n)
# solve the problem using the sparse backend
# and suppose you want to change the accuracy to 1.E-9 and rho initial value to 1.E-7
results = proxsuite.proxqp.dense.solve(
    H=H, g=g, A=A, b=b, C=C, l=l, u=u, rho=1.0e-7, eps_abs=1.0e-9
)
# Note that in python the order does not matter for rho and eps_abs
# print an optimal solution
print("optimal x: {}".format(results.x))
print("optimal y: {}".format(results.y))
print("optimal z: {}".format(results.z))
