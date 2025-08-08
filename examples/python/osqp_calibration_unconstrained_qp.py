import proxsuite
import osqp

import numpy as np
import scipy.sparse as spa
from util import unconstrained_qp, infty_norm, status_to_string


def solve_unconstrained_qp(
    dim: int,
    verbose_solver: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
    verbose_timings: bool = False,
    adaptive_mu: bool = False,
    polishing: bool = False,
    max_iter: int = 4000,
    compute_preconditioner: bool = True,
):
    # Precision (OSQP)
    eps_abs = 1e-3
    eps_rel = 0

    # Generate a qp problem
    sparsity_factor = 0.45
    strong_convexity_factor = 1e-2

    H, g, A, b, C, u, l = unconstrained_qp(
        dim, sparsity_factor, strong_convexity_factor
    )

    # OSQP proxsuite
    proxsuite_osqp = proxsuite.osqp.dense.QP(dim, 0, 0)
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
    x_source = res_source.x

    r_dua_proxsuite = proxsuite_osqp.results.info.dua_res
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

    # Prints calibration OSQP proxsuite vs source
    if verbose_results_variables or verbose_calibration:
        print("-----------------------------------------------------------------")
        print("")
        print("Comparison of results between OSQP proxsuite and source")

    if verbose_results_variables:
        print("")
        print("x")
        print("OSQP proxsuite")
        print(x_proxsuite)
        print("OSQP source")
        print(x_source)

    if verbose_calibration:
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

    # Calibration results
    cal_res = {
        "x_proxsuite": x_proxsuite,
        "x_source": x_source,
        "r_dua_proxsuite": r_dua_proxsuite,
        "r_dua_source": r_dua_source,
        "iter_proxsuite": iter_proxsuite,
        "iter_source": iter_source,
        "status_proxsuite": status_proxsuite,
        "status_source": status_source,
    }
    return cal_res


# Test calibration
def test_calibration_unconstrained_qp(
    dim_start: int = 10,
    dim_end: int = 1000,
    dim_step: int = 100,
    verbose_solver: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
    verbose_timings: bool = False,
    adaptive_mu: bool = False,
    polishing: bool = False,
    max_iter: int = 4000,
    compute_preconditioner: bool = True,
    prec_x: float = 1e-3,
    prec_r_dua: float = 1e-3,
):
    # Diff lists
    diff_x_lst = []
    diff_r_dua_lst = []
    diff_iter_lst = []
    diff_status_lst = []

    nb_tests = 0
    failed_tests = 0

    for dim in range(dim_start, dim_end, dim_step):
        cal_res = solve_unconstrained_qp(
            dim=dim,
            verbose_solver=verbose_solver,
            verbose_results_variables=verbose_results_variables,
            verbose_calibration=verbose_calibration,
            verbose_timings=verbose_timings,
            adaptive_mu=adaptive_mu,
            polishing=polishing,
            max_iter=max_iter,
            compute_preconditioner=compute_preconditioner,
        )

        x_proxsuite = cal_res["x_proxsuite"]
        x_source = cal_res["x_source"]
        r_dua_proxsuite = cal_res["r_dua_proxsuite"]
        r_dua_source = cal_res["r_dua_source"]
        iter_proxsuite = cal_res["iter_proxsuite"]
        iter_source = cal_res["iter_source"]
        status_proxsuite = cal_res["status_proxsuite"]
        status_source = cal_res["status_source"]

        max_diff_x = infty_norm(x_proxsuite - x_source)
        same_x = max_diff_x <= prec_x

        error_r_dua = np.abs(r_dua_proxsuite - r_dua_source)
        same_r_dua = error_r_dua <= prec_r_dua

        same_iter = iter_proxsuite == iter_source

        both_succeed = (
            status_proxsuite == proxsuite.osqp.PROXQP_SOLVED
            and status_source == "solved"
        )
        both_max_iter = (
            status_proxsuite == proxsuite.osqp.PROXQP_MAX_ITER_REACHED
            and status_source == "maximum iterations reached"
        )
        same_status = True if (both_succeed or both_max_iter) else False

        if not same_x:
            print("")
            print("x differs in dim = ", dim, " at precision ", prec_x, ":")
            if dim <= 30:
                print("Proxsuite: ")
                print(x_proxsuite)
                print("Source: ")
                print(x_source)
            else:
                print("dim ", dim, " > 30 too large for visualization.")
            print("Max error: ")
            print(max_diff_x)
            diff_x_lst.append(dim)

        if not same_r_dua:
            print("")
            print("r_dua differs in dim = ", dim, " at precision ", prec_r_dua, ":")
            print("Proxsuite: ")
            print(r_dua_proxsuite)
            print("Source: ")
            print(r_dua_source)
            print("Error")
            print(error_r_dua)
            diff_r_dua_lst.append(dim)

        if not same_iter:
            print("")
            print("iter differs in dim = ", dim, ":")
            print("Proxsuite: ")
            print(iter_proxsuite)
            print("Source: ")
            print(iter_source)
            diff_iter_lst.append(dim)

        if not same_status:
            print("")
            print("status differs in dim = ", dim, ":")
            print("Proxsuite: ")
            print(status_to_string(status_proxsuite))
            print("Source: ")
            print(status_source)
            diff_status_lst.append(dim)

        if not (same_x and same_r_dua and same_iter and same_status):
            failed_tests += 1
        nb_tests += 1

    print("")
    print("Results of calibration test")
    print("Number of tests: ", nb_tests, " | Tests failed: ", failed_tests)

    print("")
    print("diff_x_lst (prec_x = ", prec_x, "):")
    print(diff_x_lst)

    print("")
    print("diff_r_dua_lst (prec_x = ", prec_r_dua, "):")
    print(diff_r_dua_lst)

    print("")
    print("diff_iter_lst:")
    print(diff_iter_lst)

    print("")
    print("diff_status_lst:")
    print(diff_status_lst)


# Run test
test_calibration_unconstrained_qp(
    dim_start=10,
    dim_end=1000,
    dim_step=20,
)

# Notes:

# => Implem very close from source
