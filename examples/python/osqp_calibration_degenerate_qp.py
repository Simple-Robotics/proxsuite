import proxsuite
import osqp

import numpy as np
import scipy.sparse as spa
from util import degenerate_qp


def solve_degenerate_qp(
    dim: int,
    n_eq: int,
    n_in: int,
    verbose_solver: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
    verbose_timings: bool = False,
    adaptive_mu: bool = True,
    polishing: bool = False,
    max_iter: int = 4000,
    compute_preconditioner: bool = True,
):
    # Generate a degenerate qp problem
    sparsity_factor = 0.45
    strong_convexity_factor = 1e-2
    eps_abs = 1e-3
    eps_rel = 0

    m = n_in // 2

    H, g, A, b, C, u, l = degenerate_qp(
        dim, n_eq, m, sparsity_factor, strong_convexity_factor
    )

    # OSQP proxsuite
    proxsuite_osqp = proxsuite.osqp.dense.QP(dim, n_eq, n_in)
    proxsuite_osqp.init(H, g, A, b, C, l, u)

    proxsuite_osqp.settings.verbose = verbose_solver
    proxsuite_osqp.settings.eps_abs = eps_abs
    proxsuite_osqp.settings.eps_rel = eps_rel

    proxsuite_osqp.settings.adaptive_mu = adaptive_mu
    proxsuite_osqp.settings.polishing = polishing

    proxsuite_osqp.settings.max_iter = max_iter
    proxsuite_osqp.settings.compute_preconditioner = compute_preconditioner

    proxsuite_osqp.solve()

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
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        sigma=1e-6,
        rho=0.1,
        verbose=verbose_solver,
        scaling=10 if compute_preconditioner else 0,
        max_iter=max_iter,
        warm_start=False,
        check_termination=1,
        adaptive_rho=adaptive_mu,
        adaptive_rho_interval=50,
        adaptive_rho_tolerance=5.0,
        polish=polishing,
    )
    res_source = prob.solve()

    # Check results
    x_proxsuite = proxsuite_osqp.results.x
    y_proxsuite = proxsuite_osqp.results.y
    z_proxsuite = proxsuite_osqp.results.z

    x_source = res_source.x
    y_source = res_source.y

    r_pri_proxsuite = proxsuite_osqp.results.info.pri_res
    r_dua_proxsuite = proxsuite_osqp.results.info.dua_res

    r_pri_source = res_source.info.pri_res
    r_dua_source = res_source.info.dua_res

    iter_proxsuite = proxsuite_osqp.results.info.iter_ext
    iter_source = res_source.info.iter

    rho_osqp_estimate_proxsuite = proxsuite_osqp.results.info.rho_osqp_estimate
    rho_osqp_estimate_source = res_source.info.rho_estimate

    mu_updates_proxsuite = proxsuite_osqp.results.info.mu_updates
    mu_updates_source = res_source.info.rho_updates

    status_proxsuite = proxsuite_osqp.results.info.status
    status_source = res_source.info.status

    setup_time_proxsuite = proxsuite_osqp.results.info.setup_time
    setup_time_source = res_source.info.setup_time

    solve_time_proxsuite = proxsuite_osqp.results.info.solve_time
    solve_time_source = res_source.info.solve_time

    run_time_proxsuite = proxsuite_osqp.results.info.run_time
    run_time_source = res_source.info.run_time

    proxsuite_pass = status_proxsuite == proxsuite.osqp.PROXQP_SOLVED
    source_pass = status_source == "solved"

    # Prints calibration OSQP proxsuite vs source
    if verbose_results_variables:
        print("")
        print("x")
        print("OSQP proxsuite")
        print(x_proxsuite)
        print("OSQP source")
        print(x_source)

        print("")
        print("(y, z) (proxsuite) vs y (source)")
        print("OSQP proxsuite")
        print(np.concatenate([y_proxsuite, z_proxsuite]))
        print("OSQP source")
        print(y_source)

    if verbose_calibration:
        print("")
        print("r_pri")
        print("OSQP proxsuite")
        print(r_pri_proxsuite)
        print("OSQP source")
        print(r_pri_source)

        print("")
        print("r_dua")
        print("OSQP proxsuite")
        print(r_dua_proxsuite)
        print("OSQP source")
        print(r_dua_source)

        print("")
        print("iter")
        print("OSQP proxsuite")
        print(iter_proxsuite)
        print("OSQP source")
        print(iter_source)

        print("")
        print("status")
        print("OSQP proxsuite")
        print(status_proxsuite)
        print("OSQP source")
        print(status_source)

        if adaptive_mu:
            print("")
            print("rho_osqp_estimate")
            print("OSQP proxsuite")
            print(rho_osqp_estimate_proxsuite)
            print("OSQP source")
            print(rho_osqp_estimate_source)

            print("")
            print("mu_updates")
            print("OSQP proxsuite")
            print(mu_updates_proxsuite)
            print("OSQP source")
            print(mu_updates_source)

        if verbose_timings:
            print("")
            print("setup_time (micro sec)")
            print("OSQP proxsuite")
            print(setup_time_proxsuite)
            print("OSQP source")
            print(1e6 * setup_time_source)

            print("")
            print("solve_time (micro sec)")
            print("OSQP proxsuite")
            print(solve_time_proxsuite)
            print("OSQP source")
            print(1e6 * solve_time_source)

            print("")
            print("run_time (micro sec)")
            print("OSQP proxsuite")
            print(run_time_proxsuite)
            print("OSQP source")
            print(1e6 * run_time_source)

    return proxsuite_pass, source_pass


source_pass_list = []
source_fail_list = []
proxsuite_pass_list = []
proxsuite_fail_list = []
for dim in range(10, 1000, 100):
    proxsuite_pass, source_pass = solve_degenerate_qp(
        dim,
        n_eq=0,
        n_in=dim // 2,
        verbose_solver=True,
        verbose_results_variables=False,
        verbose_calibration=True,
        verbose_timings=False,
        adaptive_mu=False,
        polishing=False,
        max_iter=4000,
        compute_preconditioner=True,
    )

    if proxsuite_pass:
        proxsuite_pass_list.append(dim)
    else:
        proxsuite_fail_list.append(dim)

    if source_pass:
        source_pass_list.append(dim)
    else:
        source_fail_list.append(dim)

print("")
print("Which test passed/failed on which solver")

print("")
print("proxsuite_pass_list:")
print(proxsuite_pass_list)

print("")
print("source_pass_list:")
print(source_pass_list)

print("")
print("proxsuite_fail_list:")
print(proxsuite_fail_list)

print("")
print("source_fail_list:")
print(source_fail_list)


# Results at commit: 63bf0cc981abdd11e398098a0d9722c287442264


# adaptive_mu = False
# Pass proxsuite: [10, 910]
# Fail proxsuite: [110, 210, 310, 410, 510, 610, 710, 810]

# adaptive_mu = False
# Pass source: [10, 110, 210, 310, 410, 510, 610, 710, 810, 910]
# Fail source: []


# Goal: Passes for all tests, even without mu_update

# adaptive_mu = False
# iter proxsuite / source

# Passed: i = 10: 46 / 150
# Passed: i = 910: 208 / 208

# Failed: Primal infeasible, i = 110: 78 / 508
# Failed: Primal infeasible, i = 210: 83 / 136
# Failed: Primal infeasible, i = 310: 90 / 631
# Failed: Primal infeasible, i = 410: 171 / 799
# Failed: Primal infeasible, i = 510: 125 / 171
# Failed: Primal infeasible, i = 610: 248 / 540
# Failed: Primal infeasible, i = 710: 183 / 421
# Failed: Primal infeasible, i = 810: 244 / 400

# Remarks:
# - In difficult problems, many iterations are needed (OSQP source)
# - TODO: Calibration ADMM only
