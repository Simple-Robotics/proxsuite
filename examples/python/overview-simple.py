import proxsuite
from util import generate_mixed_qp
import numpy as np

# generate a qp problem
n = 10
H = np.eye(10)
g = np.zeros(10)
A = np.zeros((1, 10))
A[0] = 1.0
b = np.zeros(1)
C = A[:]
u = np.ones(1) * -1
l = np.ones(1) * 1
n_eq = A.shape[0]
n_in = C.shape[0]

# solve it
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
qp.init(H, g, A, b, C, l, u)
qp.solve()
# print an optimal solution
print(f"optimal x: {qp.results.x}")
print(f"optimal y: {qp.results.y}")
print(f"optimal z: {qp.results.z}")
