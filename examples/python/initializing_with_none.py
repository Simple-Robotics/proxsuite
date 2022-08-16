import numpy as np
import proxsuite

H = np.array([[65.0, -22.0, -16.0], [-22.0, 14.0, 7.0], [-16.0, 7.0, 5.0]])
g = np.array([-13.0, 15.0, 7.0])
A = None
b = None
C = None
u = None
l = None

Qp = proxsuite.proxqp.dense.QP(3,0,0)
Qp.init(H, g, A, b, C, u, l) # it is equivalent to do as well Qp.init(H, g)
Qp.solve()
print("optimal x: {}".format(Qp.results.x))