import numpy as np
import proxsuite

H = np.array([[65.0, -22.0, -16.0], [-22.0, 14.0, 7.0], [-16.0, 7.0, 5.0]])
g = np.array([-13.0, 15.0, 7.0])
A = None
b = None
C = None
u = None
l = None

qp = proxsuite.proxqp.dense.QP(3, 0, 0)
qp.init(H, g, A, b, C, l, u)  # it is equivalent to do as well qp.init(H, g)
qp.solve()
print("optimal x: {}".format(qp.results.x))
