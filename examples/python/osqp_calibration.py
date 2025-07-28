import proxsuite
import osqp

import numpy as np
import scipy.sparse as spa
from util import generate_mixed_qp


# Generate a qp problem
n = 10
H, g, A, b, C, u, l = generate_mixed_qp(n)
n_eq = A.shape[0]
n_in = C.shape[0]

# OSQP proxsuite
qp = proxsuite.osqp.dense.QP(n, n_eq, n_in)
qp.init(H, g, A, b, C, l, u)
qp.solve()

# OSQP source code
H_source = spa.csc_matrix(H)
A_sparse = spa.csc_matrix(A)
C_sparse = spa.csc_matrix(C)
g_source = g
l_source = np.concatenate([b, l])
u_source = np.concatenate([b, u])
A_source = spa.vstack([A_sparse, C_sparse], format="csc")
prob = osqp.OSQP()
prob.setup(H_source, g_source, A_source, l_source, u_source, verbose=False)
res = prob.solve()

# PROXQP proxsuite
qp_proxqp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
qp_proxqp.init(H, g, A, b, C, l, u)
qp_proxqp.solve()

# Prints results
verbose_all = False
if verbose_all:
    print("Optimal x")
    print("OSQP proxsuite")
    print(qp.results.x)
    print("OSQP source")
    print(res.x)
    print("PROXQP proxsuite")
    print(qp_proxqp.results.x)

    print("")
    print("Optimal y (OSQP source) or (y, z) (proxsuite)")
    y_z_osqp = np.concatenate([qp.results.y, qp.results.z])
    y_z_proxqp = np.concatenate([qp_proxqp.results.y, qp_proxqp.results.z])
    print("OSQP proxsuite")
    print(y_z_osqp)
    print("OSQP source")
    print(res.y)
    print("PROXQP proxsuite")
    print(y_z_proxqp)

# Prints calibration OSQP proxsuite vs source
verbose_calibration = True
if verbose_calibration:
    print("")
    print("x")
    print("OSQP proxsuite")
    print(qp.results.x)
    print("OSQP source")
    print(res.x)

    print("")
    print("(y, z) (proxsuite) vs y (source)")
    y_z_osqp = np.concatenate([qp.results.y, qp.results.z])
    y_z_proxqp = np.concatenate([qp_proxqp.results.y, qp_proxqp.results.z])
    print("OSQP proxsuite")
    print(y_z_osqp)
    print("OSQP source")
    print(res.y)

    print("")
    print("iter")
    print("OSQP proxsuite")
    print(qp.results.info.iter_ext)
    print("OSQP source")
    print(res.info.iter)

    print("")
    print("mu_eq")
    print("OSQP proxsuite")
    print(qp.results.info.mu_eq)
    print("OSQP source")
    print(1e3 / res.info.rho_estimate)

    print("")
    print("mu_in")
    print("OSQP proxsuite")
    print(qp.results.info.mu_in)
    print("OSQP source")
    print(1 / res.info.rho_estimate)

    print("")
    print("mu_updates")
    print("OSQP proxsuite")
    print(qp.results.info.mu_updates)
    print("OSQP source")
    print(res.info.rho_updates)

    print("")
    print("status")
    print("OSQP proxsuite")
    print(qp.results.info.status)
    print("OSQP source")
    print(res.info.status)

    print("")
    print("setup_time")
    print("OSQP proxsuite")
    print(qp.results.info.setup_time)
    print("OSQP source")
    print(res.info.setup_time)

    print("")
    print("solve_time")
    print("OSQP proxsuite")
    print(qp.results.info.setup_time)
    print("OSQP source")
    print(res.info.setup_time)

    print("")
    print("run_time")
    print("OSQP proxsuite")
    print(qp.results.info.setup_time)
    print("OSQP source")
    print(res.info.setup_time)
