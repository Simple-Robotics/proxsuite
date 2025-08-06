import proxsuite
import osqp

import numpy as np
import scipy.sparse as spa

from util import load_qp, preprocess_qp
from pathlib import Path


def solve_maros_maszaros(
    filename: str,
    verbose: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
):
    """
    This function aims at reproducing the behaviour of the unit test
    on the Maros Meszaros problem data (OSQP Dense).

    In particular, we run both our implementation and the source code,
    with the same settings, on the same problems (only the first step
    with initial guess = NO_INITIAL_GUESS).

    Doing so, we justify the deletion of some Maros Meszaros problems
    in the unit test that would fail, not by our fault in the
    implementation, but due to the OSQP algorithm itself.

    In order to visualize the gap in performance between OSQP and
    PROXQP for example, we refer to the paper of the latter:
    https://inria.hal.science/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/
    """

    # MM problem
    qp = load_qp(filename)
    proprocessed = preprocess_qp(qp)
    H = proprocessed.H
    A = proprocessed.A
    C = proprocessed.C
    g = proprocessed.g
    b = proprocessed.b
    u = proprocessed.u
    l = proprocessed.l

    dim = H.shape[0]
    n_eq = A.shape[0]
    n_in = C.shape[0]

    eps_abs = 1e-3
    eps_rel = 0
    eps_primal_inf = 1e-12
    eps_dual_inf = 1e-12

    # OSQP proxsuite
    proxsuite_osqp = proxsuite.osqp.dense.QP(dim, n_eq, n_in, box_constraints=False)
    proxsuite_osqp.init(H, g, A, b, C, l, u)

    proxsuite_osqp.settings.verbose = verbose
    proxsuite_osqp.settings.eps_abs = eps_abs
    proxsuite_osqp.settings.eps_rel = eps_rel
    proxsuite_osqp.settings.eps_primal_inf = eps_primal_inf
    proxsuite_osqp.settings.eps_dual_inf = eps_dual_inf

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
        verbose=verbose,
        scaling=10,
        max_iter=4000,
        warm_start=False,
        check_termination=1,
        adaptive_rho=True,
        adaptive_rho_interval=50,
        adaptive_rho_tolerance=5.0,
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

    mu_eq_proxsuite = proxsuite_osqp.results.info.mu_eq
    mu_eq_source = 1e3 / res_source.info.rho_estimate

    mu_in_proxsuite = proxsuite_osqp.results.info.mu_in
    mu_in_source = 1 / res_source.info.rho_estimate

    mu_updates_proxsuite = proxsuite_osqp.results.info.mu_updates
    mu_updates_source = res_source.info.rho_updates

    status_proxsuite = proxsuite_osqp.results.info.status
    status_source = res_source.info.status

    if verbose_calibration:
        print("primal residual proxsuite :")
        print(r_pri_proxsuite)
        print("dua residual proxsuite :")
        print(r_dua_proxsuite)
        print("iter proxsuite :")
        print(iter_proxsuite)

        print("primal residual source :")
        print(r_pri_source)
        print("dua residual source :")
        print(r_dua_source)
        print("iter source :")
        print(iter_source)

    eps = proxsuite_osqp.settings.eps_abs = eps_abs

    proxsuite_pass = (
        r_pri_proxsuite > -eps
        and r_dua_proxsuite < 2 * eps
        and np.min(C @ x_proxsuite - l) > -eps
        and np.min(C @ x_proxsuite - u) < eps
    )

    source_pass = (
        r_pri_source > -eps
        and r_dua_source < 2 * eps
        and np.min(C @ x_source - l) > -eps
        and np.min(C @ x_source - u) < eps
    )

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
        print("mu_eq")
        print("OSQP proxsuite")
        print(mu_eq_proxsuite)
        print("OSQP source")
        print(mu_eq_source)

        print("")
        print("mu_in")
        print("OSQP proxsuite")
        print(mu_in_proxsuite)
        print("OSQP source")
        print(mu_in_source)

        print("")
        print("mu_updates")
        print("OSQP proxsuite")
        print(mu_updates_proxsuite)
        print("OSQP source")
        print(mu_updates_source)

        print("")
        print("status")
        print("OSQP proxsuite")
        print(status_proxsuite)
        print("OSQP source")
        print(status_source)

    return proxsuite_pass, source_pass


REPO_ROOT = Path(__file__).resolve().parents[2]
MAROS_MESZAROS_DIR = REPO_ROOT / "test" / "data" / "maros_meszaros_data"

files = [
    # Proxsuite OSQP fails in cpp unit test
    MAROS_MESZAROS_DIR / "PRIMALC1.mat",
    MAROS_MESZAROS_DIR / "PRIMALC2.mat",
    MAROS_MESZAROS_DIR / "PRIMALC5.mat",
    MAROS_MESZAROS_DIR / "PRIMALC8.mat",
    MAROS_MESZAROS_DIR / "QBANDM.mat",
    MAROS_MESZAROS_DIR / "QBORE3D.mat",
    MAROS_MESZAROS_DIR / "QBRANDY.mat",
    MAROS_MESZAROS_DIR / "QCAPRI.mat",
    MAROS_MESZAROS_DIR / "QE226.mat",
    MAROS_MESZAROS_DIR / "QFORPLAN.mat",
    MAROS_MESZAROS_DIR / "QFORPLAN.mat",
    MAROS_MESZAROS_DIR / "QGROW7.mat",
    MAROS_MESZAROS_DIR / "QISRAEL.mat",
    MAROS_MESZAROS_DIR / "QPCBOEI1.mat",
    MAROS_MESZAROS_DIR / "QPCBOEI2.mat",
    MAROS_MESZAROS_DIR / "QSCAGR25.mat",
    MAROS_MESZAROS_DIR / "QSCAGR7.mat",
    MAROS_MESZAROS_DIR / "QSCFXM1.mat",
    MAROS_MESZAROS_DIR / "QSCTAP1.mat",
    MAROS_MESZAROS_DIR / "QSHARE1B.mat",
    MAROS_MESZAROS_DIR / "QSHARE2B.mat",
    MAROS_MESZAROS_DIR / "QSTAIR.mat",
    # Proxsuite OSQP passes in cpp unit test
    MAROS_MESZAROS_DIR / "CVXQP1_S.mat",
    MAROS_MESZAROS_DIR / "CVXQP2_S.mat",
    MAROS_MESZAROS_DIR / "CVXQP3_S.mat",
    MAROS_MESZAROS_DIR / "DPKLO1.mat",
    MAROS_MESZAROS_DIR / "DUAL1.mat",
    MAROS_MESZAROS_DIR / "DUAL2.mat",
    MAROS_MESZAROS_DIR / "DUAL3.mat",
    MAROS_MESZAROS_DIR / "DUAL4.mat",
    MAROS_MESZAROS_DIR / "DUALC1.mat",
    MAROS_MESZAROS_DIR / "DUALC2.mat",
    MAROS_MESZAROS_DIR / "DUALC5.mat",
    MAROS_MESZAROS_DIR / "DUALC8.mat",
    MAROS_MESZAROS_DIR / "GENHS28.mat",
    MAROS_MESZAROS_DIR / "HS118.mat",
    MAROS_MESZAROS_DIR / "HS21.mat",
    MAROS_MESZAROS_DIR / "HS268.mat",
    MAROS_MESZAROS_DIR / "HS35.mat",
    MAROS_MESZAROS_DIR / "HS35MOD.mat",
    MAROS_MESZAROS_DIR / "HS51.mat",
    MAROS_MESZAROS_DIR / "HS52.mat",
    MAROS_MESZAROS_DIR / "HS53.mat",
    MAROS_MESZAROS_DIR / "HS76.mat",
    MAROS_MESZAROS_DIR / "LOTSCHD.mat",
    MAROS_MESZAROS_DIR / "PRIMAL1.mat",
    MAROS_MESZAROS_DIR / "PRIMAL2.mat",
    MAROS_MESZAROS_DIR / "PRIMAL3.mat",
    MAROS_MESZAROS_DIR / "QADLITTL.mat",
    MAROS_MESZAROS_DIR / "QAFIRO.mat",
    MAROS_MESZAROS_DIR / "QBEACONF.mat",
    MAROS_MESZAROS_DIR / "QPCBLEND.mat",
    MAROS_MESZAROS_DIR / "QSCORPIO.mat",
    MAROS_MESZAROS_DIR / "QSCSD1.mat",
    MAROS_MESZAROS_DIR / "S268.mat",
    MAROS_MESZAROS_DIR / "TAME.mat",
    MAROS_MESZAROS_DIR / "VALUES.mat",
    MAROS_MESZAROS_DIR / "ZECEVIC2.mat",
    MAROS_MESZAROS_DIR / "QPCSTAIR.mat",
]

both_pass = []
both_fail = []
proxsuite_pass_source_fail = []
source_pass_proxsuite_fail = []
for file in files:
    filename = str(file)
    proxsuite_pass, source_pass = solve_maros_maszaros(
        filename,
        verbose=True,
        verbose_results_variables=False,
        verbose_calibration=True,
    )

    filename_only = Path(filename).name

    if proxsuite_pass and source_pass:
        both_pass.append(filename_only)

    elif proxsuite_pass and not source_pass:
        proxsuite_pass_source_fail.append(filename_only)

    elif not proxsuite_pass and source_pass:
        source_pass_proxsuite_fail.append(filename_only)

    elif not proxsuite_pass and not source_pass:
        both_fail.append(filename_only)

print("")
print("Which test passed/failed on which solver")

print("")
print("both_pass:")
print(both_pass)

print("")
print("both_fail:")
print(both_fail)

print("")
print("proxsuite_pass_source_fail:")
print(proxsuite_pass_source_fail)

print("")
print("source_pass_proxsuite_fail:")
print(source_pass_proxsuite_fail)
