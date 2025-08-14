import proxsuite
import osqp

import numpy as np
import scipy.sparse as spa

from utils import load_qp, preprocess_qp
from pathlib import Path


def solve_maros_maszaros(
    filename: str,
    verbose_solver: bool = False,
    verbose_results_variables: bool = False,
    verbose_calibration: bool = False,
):
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

    proxsuite_osqp.settings.verbose = verbose_solver
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
        verbose=verbose_solver,
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
        print("status")
        print("OSQP proxsuite")
        print(status_proxsuite)
        print("OSQP source")
        print(status_source)

    return proxsuite_pass, source_pass


def test_calibration_maros_meszaros(
    test_skipped_problems: bool = False,
    verbose_solver=False,
    verbose_results_variables=False,
    verbose_calibration=False,
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

    Args:
    - test_skipped_problems: Boolean to deceide whether we test on the problems
    that are skipped in the unit test, due to high dimensionality. It filters
    data with dim > 1000 or n_eq + n_in > 1000.
    """

    REPO_ROOT = Path(__file__).resolve().parents[3]
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
    if test_skipped_problems:
        files_skipped = [
            # Skipped problems in unit test
            MAROS_MESZAROS_DIR / "AUG2D.mat",
            MAROS_MESZAROS_DIR / "AUG2DC.mat",
            MAROS_MESZAROS_DIR / "AUG2DCQP.mat",
            MAROS_MESZAROS_DIR / "AUG2DQP.mat",
            MAROS_MESZAROS_DIR / "AUG3D.mat",
            MAROS_MESZAROS_DIR / "AUG3DC.mat",
            MAROS_MESZAROS_DIR / "AUG3DCQP.mat",
            MAROS_MESZAROS_DIR / "AUG3DQP.mat",
            MAROS_MESZAROS_DIR / "BOYD1.mat",
            MAROS_MESZAROS_DIR / "BOYD2.mat",
            MAROS_MESZAROS_DIR / "CONT-050.mat",
            MAROS_MESZAROS_DIR / "CONT-100.mat",
            MAROS_MESZAROS_DIR / "CONT-101.mat",
            MAROS_MESZAROS_DIR / "CONT-200.mat",
            MAROS_MESZAROS_DIR / "CONT-201.mat",
            MAROS_MESZAROS_DIR / "CONT-300.mat",
            MAROS_MESZAROS_DIR / "CVXQP1_L.mat",
            MAROS_MESZAROS_DIR / "CVXQP1_M.mat",
            MAROS_MESZAROS_DIR / "CVXQP2_L.mat",
            MAROS_MESZAROS_DIR / "CVXQP2_M.mat",
            MAROS_MESZAROS_DIR / "CVXQP3_L.mat",
            MAROS_MESZAROS_DIR / "CVXQP3_M.mat",
            MAROS_MESZAROS_DIR / "DTOC3.mat",
            MAROS_MESZAROS_DIR / "EXDATA.mat",
            MAROS_MESZAROS_DIR / "GOULDQP2.mat",
            MAROS_MESZAROS_DIR / "GOULDQP3.mat",
            MAROS_MESZAROS_DIR / "HUES-MOD.mat",
            MAROS_MESZAROS_DIR / "HUESTIS.mat",
            MAROS_MESZAROS_DIR / "KSIP.mat",
            MAROS_MESZAROS_DIR / "LASER.mat",
            MAROS_MESZAROS_DIR / "LISWET1.mat",
            MAROS_MESZAROS_DIR / "LISWET10.mat",
            MAROS_MESZAROS_DIR / "LISWET11.mat",
            MAROS_MESZAROS_DIR / "LISWET12.mat",
            MAROS_MESZAROS_DIR / "LISWET2.mat",
            MAROS_MESZAROS_DIR / "LISWET3.mat",
            MAROS_MESZAROS_DIR / "LISWET4.mat",
            MAROS_MESZAROS_DIR / "LISWET5.mat",
            MAROS_MESZAROS_DIR / "LISWET6.mat",
            MAROS_MESZAROS_DIR / "LISWET7.mat",
            MAROS_MESZAROS_DIR / "LISWET8.mat",
            MAROS_MESZAROS_DIR / "LISWET9.mat",
            MAROS_MESZAROS_DIR / "MOSARQP1.mat",
            MAROS_MESZAROS_DIR / "MOSARQP2.mat",
            MAROS_MESZAROS_DIR / "POWELL20.mat",
            MAROS_MESZAROS_DIR / "PRIMAL4.mat",
            MAROS_MESZAROS_DIR / "Q25FV47.mat",
            MAROS_MESZAROS_DIR / "QETAMACR.mat",
            MAROS_MESZAROS_DIR / "QFFFFF80.mat",
            MAROS_MESZAROS_DIR / "QGFRDXPN.mat",
            MAROS_MESZAROS_DIR / "QGROW22.mat",
            MAROS_MESZAROS_DIR / "QPILOTNO.mat",
            MAROS_MESZAROS_DIR / "QPTEST.mat",
            MAROS_MESZAROS_DIR / "QRECIPE.mat",
            MAROS_MESZAROS_DIR / "QSC205.mat",
            MAROS_MESZAROS_DIR / "QSCFXM2.mat",
            MAROS_MESZAROS_DIR / "QSCFXM3.mat",
            MAROS_MESZAROS_DIR / "QSCRS8.mat",
            MAROS_MESZAROS_DIR / "QSCSD6.mat",
            MAROS_MESZAROS_DIR / "QSCSD8.mat",
            MAROS_MESZAROS_DIR / "QSCTAP2.mat",
            MAROS_MESZAROS_DIR / "QSCTAP3.mat",
            MAROS_MESZAROS_DIR / "QSEBA.mat",
            MAROS_MESZAROS_DIR / "QSHELL.mat",
            MAROS_MESZAROS_DIR / "QSHIP04L.mat",
            MAROS_MESZAROS_DIR / "QSHIP04S.mat",
            MAROS_MESZAROS_DIR / "QSHIP08L.mat",
            MAROS_MESZAROS_DIR / "QSHIP08S.mat",
            MAROS_MESZAROS_DIR / "QSHIP12L.mat",
            MAROS_MESZAROS_DIR / "QSHIP12S.mat",
            MAROS_MESZAROS_DIR / "QSIERRA.mat",
            MAROS_MESZAROS_DIR / "QSTANDAT.mat",
            MAROS_MESZAROS_DIR / "STADAT1.mat",
            MAROS_MESZAROS_DIR / "STADAT2.mat",
            MAROS_MESZAROS_DIR / "STADAT3.mat",
            MAROS_MESZAROS_DIR / "STCQP1.mat",
            MAROS_MESZAROS_DIR / "STCQP2.mat",
            MAROS_MESZAROS_DIR / "UBH1.mat",
            MAROS_MESZAROS_DIR / "YAO.mat",
        ]
        files = files + files_skipped

    both_pass = []
    both_fail = []
    proxsuite_pass_source_fail = []
    source_pass_proxsuite_fail = []
    for file in files:
        filename = str(file)
        proxsuite_pass, source_pass = solve_maros_maszaros(
            filename,
            verbose_solver=verbose_solver,
            verbose_results_variables=verbose_results_variables,
            verbose_calibration=verbose_calibration,
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


# test_calibration_maros_meszaros(
#     test_skipped_problems=True,
#     verbose_solver=True,
#     verbose_results_variables=False,
#     verbose_calibration=False,
# )
