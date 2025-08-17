import proxsuite
import osqp

import numpy as np
import scipy.sparse as spa

from utils import infty_norm, status_to_string, status_polish_to_string
from utils import (
    unconstrained_qp,
    strongly_convex_qp,
    not_strongly_convex_qp,
    degenerate_qp,
    box_constrained_qp,
    primal_infeasible_qp,
    dual_infeasible_qp,
)


def solve_qp(
    problem: str,
    dim: int,
    n_eq: int,
    n_in: int,
    m: int,
    max_iter: int = 4000,
    compute_preconditioner: bool = True,
    eps_abs: float = 1e-3,
    eps_rel: float = 0,
    eps_primal_inf: float = 1e-4,
    eps_dual_inf: float = 1e-4,
    sparsity_factor: float = 0.45,
    strong_convexity_factor: float = 1e-2,
    adaptive_mu: bool = False,
    adaptive_mu_interval: int = 50,
    adaptive_mu_tolerance: float = 5.0,
    polish: bool = False,
    delta: float = 1e-6,
    polish_refine_iter: int = 3,
    verbose_solver: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
    verbose_timings: bool = False,
):
    if problem == "unconstrained_qp":
        H, g, A, b, C, u, l = unconstrained_qp(
            dim, sparsity_factor, strong_convexity_factor
        )
    elif problem == "strongly_convex_qp":
        H, g, A, b, C, u, l = strongly_convex_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor
        )
    elif problem == "not_strongly_convex_qp":
        H, g, A, b, C, u, l = not_strongly_convex_qp(dim, n_eq, n_in, sparsity_factor)
    elif problem == "degenerate_qp":
        H, g, A, b, C, u, l = degenerate_qp(
            dim, n_eq, m, sparsity_factor, strong_convexity_factor
        )
    elif problem == "box_constrained_qp":
        H, g, A, b, C, u, l = box_constrained_qp(
            dim, n_eq, sparsity_factor, strong_convexity_factor
        )
    elif problem == "primal_infeasible_qp":
        H, g, A, b, C, u, l = primal_infeasible_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor
        )
    elif problem == "dual_infeasible_qp":
        H, g, A, b, C, u, l = dual_infeasible_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor
        )

    # OSQP proxsuite
    proxsuite_osqp = proxsuite.osqp.dense.QP(dim, n_eq, n_in)
    proxsuite_osqp.init(H, g, A, b, C, l, u)

    proxsuite_osqp.settings.verbose = verbose_solver

    proxsuite_osqp.settings.eps_abs = eps_abs
    proxsuite_osqp.settings.eps_rel = eps_rel

    proxsuite_osqp.settings.eps_primal_inf = eps_primal_inf
    proxsuite_osqp.settings.eps_dual_inf = eps_dual_inf

    proxsuite_osqp.settings.adaptive_mu = adaptive_mu
    proxsuite_osqp.settings.adaptive_mu_interval = adaptive_mu_interval
    proxsuite_osqp.settings.adaptive_mu_tolerance = adaptive_mu_tolerance

    proxsuite_osqp.settings.polish = polish
    proxsuite_osqp.settings.delta = delta
    proxsuite_osqp.settings.polish_refine_iter = polish_refine_iter

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
        eps_prim_inf=eps_primal_inf,
        eps_dual_inf=eps_dual_inf,
        sigma=1e-6,
        rho=0.1,
        verbose=verbose_solver,
        scaling=10 if compute_preconditioner else 0,
        max_iter=max_iter,
        warm_start=False,
        check_termination=1,
        adaptive_rho=adaptive_mu,
        adaptive_rho_interval=adaptive_mu_interval,
        adaptive_rho_tolerance=adaptive_mu_tolerance,
        polish=polish,
        delta=delta,
        polish_refine_iter=polish_refine_iter,
    )
    res_source = prob.solve()

    # Check results
    x_proxsuite = proxsuite_osqp.results.x
    x_source = res_source.x

    y_proxsuite = proxsuite_osqp.results.y
    z_proxsuite = proxsuite_osqp.results.z
    yz_proxsuite = np.concatenate([y_proxsuite, z_proxsuite])
    y_source = res_source.y

    r_pri_proxsuite = proxsuite_osqp.results.info.pri_res
    r_pri_source = res_source.info.pri_res

    r_dua_proxsuite = proxsuite_osqp.results.info.dua_res
    r_dua_source = res_source.info.dua_res

    iter_proxsuite = proxsuite_osqp.results.info.iter_ext
    iter_source = res_source.info.iter

    mu_updates_proxsuite = proxsuite_osqp.results.info.mu_updates
    mu_updates_source = res_source.info.rho_updates

    status_proxsuite = proxsuite_osqp.results.info.status
    status_source = res_source.info.status

    status_polish_proxsuite = proxsuite_osqp.results.info.status_polish
    status_polish_source = res_source.info.status_polish

    setup_time_proxsuite = proxsuite_osqp.results.info.setup_time
    setup_time_source = res_source.info.setup_time

    solve_time_proxsuite = proxsuite_osqp.results.info.solve_time
    solve_time_source = res_source.info.solve_time

    run_time_proxsuite = proxsuite_osqp.results.info.run_time
    run_time_source = res_source.info.run_time

    # Prints calibration OSQP proxsuite vs source
    if verbose_results_variables or verbose_calibration:
        print("-----------------------------------------------------------------")
        print("Comparison of results between OSQP proxsuite and source")
        print("")

    if verbose_results_variables:
        print("x")
        print("OSQP proxsuite")
        print(x_proxsuite)
        print("OSQP source")
        print(x_source)
        print("")

        print("(y, z) (proxsuite) vs y (source)")
        print("OSQP proxsuite")
        print(yz_proxsuite)
        print("OSQP source")
        print(y_source)
        print("")

    if verbose_calibration:
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
        print(status_to_string(status_proxsuite, "proxsuite"))
        print("OSQP source")
        print(status_to_string(status_source, "source"))

        if adaptive_mu:
            print("mu_updates")
            print("OSQP proxsuite")
            print(mu_updates_proxsuite)
            print("OSQP source")
            print(mu_updates_source)
            print("")

        if polish:
            print("status_polish")
            print("OSQP proxsuite")
            print(status_polish_to_string(status_polish_proxsuite, "proxsuite"))
            print("OSQP source")
            print(status_polish_to_string(status_polish_source, "source"))
            print("")

        if verbose_timings:
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
            print("")

    # Calibration results
    cal_res = {
        "x_proxsuite": x_proxsuite,
        "x_source": x_source,
        "yz_proxsuite": yz_proxsuite,
        "y_source": y_source,
        "r_pri_proxsuite": r_pri_proxsuite,
        "r_pri_source": r_pri_source,
        "r_dua_proxsuite": r_dua_proxsuite,
        "r_dua_source": r_dua_source,
        "iter_proxsuite": iter_proxsuite,
        "iter_source": iter_source,
        "mu_updates_proxsuite": mu_updates_proxsuite,
        "mu_updates_source": mu_updates_source,
        "status_proxsuite": status_proxsuite,
        "status_source": status_source,
        "status_polish_proxsuite": status_polish_proxsuite,
        "status_polish_source": status_polish_source,
    }

    return cal_res


def test_calibration_qp(
    problem: str,
    dim_start: int = 10,
    dim_end: int = 1000,
    dim_step: int = 20,
    only_eq: bool = False,
    only_in: bool = False,
    max_iter: int = 4000,
    compute_preconditioner: bool = True,
    eps_abs: float = 1e-3,
    eps_rel: float = 0,
    eps_primal_inf: float = 1e-4,
    eps_dual_inf: float = 1e-4,
    sparsity_factor: float = 0.45,
    strong_convexity_factor: float = 1e-2,
    adaptive_mu: bool = False,
    adaptive_mu_interval: int = 50,
    adaptive_mu_tolerance: float = 5.0,
    polish: bool = False,
    delta: float = 1e-6,
    polish_refine_iter: int = 3,
    verbose_test_settings: bool = False,
    verbose_solver: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
    verbose_timings: bool = False,
    prec_x: float = 1e-3,
    prec_yz: float = 1e-3,
    prec_r_pri: float = 1e-3,
    prec_r_dua: float = 1e-3,
    prec_polish: float = 1e-9,
    prec_iter: int = 0,
    prec_mu_updates: int = 0,
):
    # Constraints setting
    if only_eq and only_in:
        print("only_eq and only_in cannot be set together")
        return

    if verbose_test_settings:
        function_args = locals().copy()
        print("Calibration test settings:")
        for param_name, param_value in function_args.items():
            print(f"  {param_name}: {param_value}")
        print("")

    # Diff lists and failed tests
    diff_x_lst = []
    diff_yz_lst = []
    diff_r_pri_lst = []
    diff_r_dua_lst = []
    diff_iter_lst = []
    diff_mu_updates_lst = []
    diff_status_lst = []
    diff_status_polish_lst = []

    nb_tests = 0
    failed_tests = 0

    for dim in range(dim_start, dim_end, dim_step):
        if problem in [
            "strongly_convex_qp",
            "not_strongly_convex_qp",
            "primal_infeasible_qp",
            "dual_infeasible_qp",
        ]:
            if only_eq:
                n_eq = dim // 2
                n_in = 0
            elif only_in:
                n_in = dim // 2
                n_eq = 0
            else:
                n_eq = dim // 4
                n_in = dim // 4
            m = 0

        elif problem == "unconstrained_qp":
            n_eq = 0
            n_in = 0
            m = 0

        elif problem == "degenerate_qp":
            if only_eq:
                n_eq = dim // 2
                n_in = 0
                m = 0
            elif only_in:
                m = dim // 4
                n_in = 2 * m
                n_eq = 0
            else:
                m = dim // 4
                n_in = 2 * m
                n_eq = dim // 4

        elif problem == "box_constrained_qp":
            if only_eq:
                print("only_eq makes no sense in box_constrained_qp")
                return
            elif only_in:
                n_eq = 0
                n_in = dim
            else:
                n_eq = dim // 4
                n_in = dim
            m = 0

        cal_res = solve_qp(
            problem=problem,
            dim=dim,
            n_eq=n_eq,
            n_in=n_in,
            m=m,
            max_iter=max_iter,
            compute_preconditioner=compute_preconditioner,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_primal_inf=eps_primal_inf,
            eps_dual_inf=eps_dual_inf,
            sparsity_factor=sparsity_factor,
            strong_convexity_factor=strong_convexity_factor,
            adaptive_mu=adaptive_mu,
            adaptive_mu_interval=adaptive_mu_interval,
            adaptive_mu_tolerance=adaptive_mu_tolerance,
            polish=polish,
            delta=delta,
            polish_refine_iter=polish_refine_iter,
            verbose_solver=verbose_solver,
            verbose_results_variables=verbose_results_variables,
            verbose_calibration=verbose_calibration,
            verbose_timings=verbose_timings,
        )

        x_proxsuite = cal_res["x_proxsuite"]
        x_source = cal_res["x_source"]
        yz_proxsuite = cal_res["yz_proxsuite"]
        y_source = cal_res["y_source"]
        r_pri_proxsuite = cal_res["r_pri_proxsuite"]
        r_pri_source = cal_res["r_pri_source"]
        r_dua_proxsuite = cal_res["r_dua_proxsuite"]
        r_dua_source = cal_res["r_dua_source"]
        iter_proxsuite = cal_res["iter_proxsuite"]
        iter_source = cal_res["iter_source"]
        mu_updates_proxsuite = cal_res["mu_updates_proxsuite"]
        mu_updates_source = cal_res["mu_updates_source"]
        status_proxsuite = cal_res["status_proxsuite"]
        status_source = cal_res["status_source"]
        status_polish_proxsuite = cal_res["status_polish_proxsuite"]
        status_polish_source = cal_res["status_polish_source"]

        status_proxsuite_str = status_to_string(status_proxsuite, "proxsuite")
        status_source_str = status_to_string(status_source, "source")

        same_status = status_proxsuite_str == status_source_str

        status_polish_proxsuite_str = status_polish_to_string(
            status_polish_proxsuite, "proxsuite"
        )
        status_polish_source_str = status_polish_to_string(
            status_polish_source, "source"
        )

        same_status_polish = status_polish_proxsuite_str == status_polish_source_str

        error_iter = np.abs(iter_proxsuite - iter_source)
        same_iter = error_iter <= prec_iter

        error_mu_updates = np.abs(mu_updates_proxsuite - mu_updates_source)
        same_mu_updates = error_mu_updates <= prec_mu_updates

        same_x = True
        same_yz = True
        same_r_pri = True
        same_r_dua = True

        same_pol_success = (
            same_status_polish and status_polish_source_str == "Polishing: succeeded"
        )

        eps_x = prec_x
        eps_yz = prec_yz
        eps_r_pri = prec_r_pri
        eps_r_dua = prec_r_dua

        if same_pol_success:
            eps_x = prec_polish
            eps_yz = prec_polish
            eps_r_pri = prec_polish
            eps_r_dua = prec_polish

        # Prevent x_source or y_source = [None, None, None, ...]
        if not (
            status_source == "primal infeasible" or status_source == "dual infeasible"
        ):
            error_x = infty_norm(x_proxsuite - x_source)
            same_x = error_x <= eps_x

            error_yz = infty_norm(yz_proxsuite - y_source)
            same_yz = error_yz <= eps_yz

            error_r_pri = np.abs(r_pri_proxsuite - r_pri_source)
            same_r_pri = error_r_pri <= eps_r_pri

            error_r_dua = np.abs(r_dua_proxsuite - r_dua_source)
            same_r_dua = error_r_dua <= eps_r_dua

        if not (
            status_source == "primal infeasible" or status_source == "dual infeasible"
        ):
            if not same_x:
                print("x differs in dim = ", dim, " at precision ", eps_x, ":")
                if dim <= 30:
                    print("Proxsuite: ")
                    print(x_proxsuite)
                    print("Source: ")
                    print(x_source)
                else:
                    print("dim ", dim, " > 30 too large for visualization.")
                print("Max error: ")
                print(error_x)
                print("")
                diff_x_lst.append(dim)

            if not same_yz:
                print("yz differs in dim = ", dim, " at precision ", eps_yz, ":")
                if n_eq + n_in <= 30:
                    print("Proxsuite: ")
                    print(yz_proxsuite)
                    print("Source: ")
                    print(y_source)
                else:
                    print(
                        "n_eq + n_in ",
                        n_eq + n_in,
                        " > 30 too large for visualization.",
                    )
                print("Max error: ")
                print(error_yz)
                print("")
                diff_yz_lst.append(dim)

            if not same_r_pri:
                print("r_pri differs in dim = ", dim, " at precision ", eps_r_pri, ":")
                print("Proxsuite: ")
                print(r_pri_proxsuite)
                print("Source: ")
                print(r_pri_source)
                print("Error")
                print(error_r_pri)
                print("")
                diff_r_pri_lst.append(dim)

            if not same_r_dua:
                print("r_dua differs in dim = ", dim, " at precision ", eps_r_dua, ":")
                print("Proxsuite: ")
                print(r_dua_proxsuite)
                print("Source: ")
                print(r_dua_source)
                print("Error")
                print(error_r_dua)
                print("")
                diff_r_dua_lst.append(dim)

        if not same_iter:
            print("iter differs in dim = ", dim, " at precision ", prec_iter, ":")
            print("Proxsuite: ")
            print(iter_proxsuite)
            print("Source: ")
            print(iter_source)
            print("Error")
            print(error_iter)
            print("")
            diff_iter_lst.append(dim)

        if not same_mu_updates:
            print(
                "mu_updates differs in dim = ",
                dim,
                " at precision ",
                prec_mu_updates,
                ":",
            )
            print("Proxsuite: ")
            print(mu_updates_proxsuite)
            print("Source: ")
            print(mu_updates_source)
            print("Error")
            print(error_mu_updates)
            print("")
            diff_mu_updates_lst.append(dim)

        if not same_status:
            print("status differs in dim = ", dim, ":")
            print("Proxsuite: ")
            print(status_proxsuite_str)
            print("Source: ")
            print(status_source_str)
            diff_status_lst.append(dim)
            print("")

        if not same_status_polish:
            print("status_polish differs in dim = ", dim, ":")
            print("Proxsuite: ")
            print(status_polish_to_string(status_polish_proxsuite, "proxsuite"))
            print("Source: ")
            print(status_polish_to_string(status_polish_source, "source"))
            diff_status_polish_lst.append(dim)
            print("")

        if not (
            same_x
            and same_yz
            and same_r_pri
            and same_r_dua
            and same_iter
            and same_mu_updates
            and same_status
            and same_status_polish
        ):
            failed_tests += 1
        nb_tests += 1

    print("Results of calibration test")
    print("Number of tests: ", nb_tests, " | Tests failed: ", failed_tests)
    print("")

    if polish:
        print(
            "diff criteria at a given dim:\n",
            "prec_polish =",
            prec_polish,
            "if both solvers gave results with polishing, \n else prec_x =",
            prec_x,
            ", prec_yz =",
            prec_yz,
            ", prec_r_pri =",
            prec_r_pri,
            ", prec_r_dua =",
            prec_r_dua,
        )
        print("")
    else:
        print(
            "diff criteria at a given dim:\n",
            " prec_x =",
            prec_x,
            ", prec_yz =",
            prec_yz,
            ", prec_r_pri =",
            prec_r_pri,
            ", prec_r_dua =",
            prec_r_dua,
        )
        print("")

    print("  diff_x_lst:")
    print(" ", diff_x_lst)
    print("")

    print("  diff_yz_lst:")
    print(" ", diff_yz_lst)
    print("")

    print("  diff_r_pri_lst:")
    print(" ", diff_r_pri_lst)
    print("")

    print("  diff_r_dua_lst:")
    print(" ", diff_r_dua_lst)
    print("")

    print("  diff_iter_lst:")
    print(" ", diff_iter_lst)
    print("")

    print("  diff_status_lst:")
    print(" ", diff_status_lst)
    print("")

    if adaptive_mu:
        print("  diff_mu_updates_lst:")
        print(" ", diff_mu_updates_lst)
        print("")

    if polish:
        print("  diff_status_polish_lst:")
        print(" ", diff_status_polish_lst)
        print("")
