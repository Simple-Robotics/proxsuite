import proxsuite
from util import generate_mixed_qp

# load a qp object using qp problem dimensions
n = 10
n_eq = 2
n_in = 2
qp = proxsuite.proxqp.sparse.QP(n, n_eq, n_in)


# load a qp2 object using matrix masks
H, g, A, b, C, u, l = generate_mixed_qp(n, True)

H_ = H != 0.0
A_ = A != 0.0
C_ = C != 0.0
qp2 = proxsuite.proxqp.sparse.QP(H_, A_, C_)
