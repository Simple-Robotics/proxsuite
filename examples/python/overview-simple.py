import sys
import proxsuite
from util import generate_mixed_qp
import numpy as np

# generate a qp problem
n = 10
H, g, A, b, C, u, l = generate_mixed_qp(n)
H = np.eye(10)
g = np.zeros(10)
A = np.zeros((2, 10))
A[0] = 1.0
b = np.zeros(2)
C = A[:]
l = np.ones(2) * -1
u = np.ones(2) * 1

H = np.array(
    [
        [1.35833528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05404331, 0.0],
        [0.0, 1.35833528, 0.0, 0.0, 0.0, 0.0, 0.0, -1.11274754, -0.06365157, 0.0],
        [0.0, 0.0, 1.35833528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.69609101, 0.0],
        [0.0, 0.0, 0.0, 2.31797205, 0.0, 0.0, -0.63411468, 0.0, 0.0, 0.34927708],
        [0.0, 0.0, 0.0, 0.0, 1.35833528, 0.0, 0.0, 0.0, 0.50763146, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.35833528, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.63411468, 0.0, 0.0, 1.35833528, 0.0, 0.0, 0.0],
        [0.0, -1.11274754, 0.0, 0.0, 0.0, 0.0, 0.0, 1.35833528, 0.0, 0.0],
        [
            0.05404331,
            -0.06365157,
            0.69609101,
            0.0,
            0.50763146,
            0.0,
            0.0,
            0.0,
            1.35833528,
            0.0,
        ],
        [0.0, 0.0, 0.0, 0.34927708, 0.0, 0.0, 0.0, 0.0, 0.0, 1.35833528],
    ]
)
g = np.array(
    [
        1.92383191,
        0.99109127,
        0.3806108,
        -0.74776636,
        -1.3176775,
        1.33291008,
        -0.68454025,
        1.25598233,
        0.49672635,
        0.3292758,
    ]
)
# A = np.array(
#     [
#         [0.0, 0.0, 0.11900865, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )
# b = np.array([0.14268159, 0.0])
A = np.zeros((0, 0))
b = np.array([])
C = np.array(
    [
        [0.0, 0.0, 0.0, 0.41005165, 0.0, 0.0, 0.0, 0.37756379, 0.0, 0.0],
        [0.19829972, 0.0, 0.18656139, 0.0, 0.0, 0.0, 0.0, 0.0, -0.67066229, 0.0],
    ]
)
u = np.array([0.1051245, 0.4784386])
l = np.array([-1.0e20, -1.0e20])


n_eq = A.shape[0]
n_in = C.shape[0]


print(f"{H=}", file=sys.stderr)
print(f"{g=}", file=sys.stderr)
print(f"{A=}", file=sys.stderr)
print(f"{b=}", file=sys.stderr)
print(f"{C=}", file=sys.stderr)
print(f"{u=}", file=sys.stderr)
print(f"{l=}", file=sys.stderr)
print(f"{A.shape}", file=sys.stderr)
print(f"{C.shape}", file=sys.stderr)

# solve it
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
qp.init(H, g, A, b, C, l, u)
qp.solve()
# print an optimal solution
print(f"optimal x: {qp.results.x}", file=sys.stderr)
print(f"optimal y: {qp.results.y}", file=sys.stderr)
print(f"optimal z: {qp.results.z}", file=sys.stderr)
