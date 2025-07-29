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
res_proxsuite = proxsuite.osqp.dense.solve(
    H,
    g,
    A,
    b,
    C,
    l,
    u,
    eps_abs=1e-3,
    eps_rel=1e-3,
    rho=1e-6,
    mu_eq=1e-2,
    mu_in=1e1,
    verbose=True,
    compute_preconditioner=True,
    compute_timings=True,
    max_iter=4000,
    check_duality_gap=False,
)

# OSQP source code
H_source = spa.csc_matrix(H)
A_sparse = spa.csc_matrix(A)
C_sparse = spa.csc_matrix(C)
g_source = g
l_source = np.concatenate([b, l])
u_source = np.concatenate([b, u])
A_source = spa.vstack([A_sparse, C_sparse], format="csc")

prob = osqp.OSQP()
prob.setup(
    H_source,
    g_source,
    A_source,
    l_source,
    u_source,
    eps_abs=1e-3,
    eps_rel=1e-3,
    sigma=1e-6,
    rho=0.1,
    verbose=True,
    scaling=10,
    max_iter=4000,
    warm_start=False,
    check_termination=1,
    adaptive_rho=False,
)
res_source = prob.solve()

# PROXQP proxsuite
res_proxqp = proxsuite.proxqp.dense.solve(
    H,
    g,
    A,
    b,
    C,
    l,
    u,
    eps_abs=1e-3,
    eps_rel=1e-3,
    rho=1e-6,
    mu_eq=1e-3,
    mu_in=1e-1,
    verbose=False,
    compute_preconditioner=True,
    compute_timings=True,
    max_iter=10000,
    check_duality_gap=False,
)

# Prints results
verbose_all = False
if verbose_all:
    print("Optimal x")
    print("OSQP proxsuite")
    print(res_proxsuite.x)
    print("OSQP source")
    print(res_source.x)
    print("PROXQP proxsuite")
    print(res_proxqp.x)

    print("")
    print("Optimal y (OSQP source) or (y, z) (proxsuite)")
    y_z_osqp = np.concatenate([res_proxsuite.y, res_proxsuite.z])
    y_z_proxqp = np.concatenate([res_proxqp.y, res_proxqp.z])
    print("OSQP proxsuite")
    print(y_z_osqp)
    print("OSQP source")
    print(res_source.y)
    print("PROXQP proxsuite")
    print(y_z_proxqp)

# Prints calibration OSQP proxsuite vs source
verbose_calibration = True
if verbose_calibration:
    print("")
    print("x")
    print("OSQP proxsuite")
    print(res_proxsuite.x)
    print("OSQP source")
    print(res_source.x)

    print("")
    print("(y, z) (proxsuite) vs y (source)")
    y_z_osqp = np.concatenate([res_proxsuite.y, res_proxsuite.z])
    print("OSQP proxsuite")
    print(y_z_osqp)
    print("OSQP source")
    print(res_source.y)

    print("")
    print("r_pri")
    print("OSQP proxsuite")
    print(res_proxsuite.info.pri_res)
    print("OSQP source")
    print(res_source.info.pri_res)

    print("")
    print("r_dua")
    print("OSQP proxsuite")
    print(res_proxsuite.info.dua_res)
    print("OSQP source")
    print(res_source.info.dua_res)

    print("")
    print("iter")
    print("OSQP proxsuite")
    print(res_proxsuite.info.iter_ext)
    print("OSQP source")
    print(res_source.info.iter)

    print("")
    print("mu_eq")
    print("OSQP proxsuite")
    print(res_proxsuite.info.mu_eq)
    print("OSQP source")
    print(1e3 / res_source.info.rho_estimate)

    print("")
    print("mu_in")
    print("OSQP proxsuite")
    print(res_proxsuite.info.mu_in)
    print("OSQP source")
    print(1 / res_source.info.rho_estimate)

    print("")
    print("mu_updates")
    print("OSQP proxsuite")
    print(res_proxsuite.info.mu_updates)
    print("OSQP source")
    print(res_source.info.rho_updates)

    print("")
    print("status")
    print("OSQP proxsuite")
    print(res_proxsuite.info.status)
    print("OSQP source")
    print(res_source.info.status)

    print("")
    print("setup_time (micro sec)")
    print("OSQP proxsuite")
    print(res_proxsuite.info.setup_time)
    print("OSQP source")
    print(1e6 * res_source.info.setup_time)

    print("")
    print("solve_time (micro sec)")
    print("OSQP proxsuite")
    print(res_proxsuite.info.solve_time)
    print("OSQP source")
    print(1e6 * res_source.info.solve_time)

    print("")
    print("run_time (micro sec)")
    print("OSQP proxsuite")
    print(res_proxsuite.info.run_time)
    print("OSQP source")
    print(1e6 * res_source.info.run_time)
