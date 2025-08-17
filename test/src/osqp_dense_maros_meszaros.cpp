//
// Copyright (c) 2025 INRIA
//
#include <doctest.hpp>
#include <maros_meszaros.hpp>
#include <proxsuite/common/utils/random_qp_problems.hpp>
#include <proxsuite/osqp/dense/dense.hpp>

using namespace proxsuite;
using namespace proxsuite::common;

#define MAROS_MESZAROS_DIR PROBLEM_PATH "/data/maros_meszaros_data/"

// Pass or fail with the settings:
// eps_abs = 1e-3, eps_rel = 0.
// adaptive_mu_update = true, adaptive_mu_interval = 50
// polish = false

// More details in /examples/python/osqp_calibration_dense_maros_meszaros.py
// Commented problems fail in both OSQP Proxsuite and source code

char const* files[] = {
  MAROS_MESZAROS_DIR "AUG2D.mat",    // Skip
  MAROS_MESZAROS_DIR "AUG2DC.mat",   // Skip
  MAROS_MESZAROS_DIR "AUG2DCQP.mat", // Skip
  MAROS_MESZAROS_DIR "AUG2DQP.mat",  // Skip
  MAROS_MESZAROS_DIR "AUG3D.mat",    // Skip
  MAROS_MESZAROS_DIR "AUG3DC.mat",   // Skip
  MAROS_MESZAROS_DIR "AUG3DCQP.mat", // Skip
  MAROS_MESZAROS_DIR "AUG3DQP.mat",  // Skip
  MAROS_MESZAROS_DIR "BOYD1.mat",    // Skip
  MAROS_MESZAROS_DIR "BOYD2.mat",    // Skip
  MAROS_MESZAROS_DIR "CONT-050.mat", // Skip
  MAROS_MESZAROS_DIR "CONT-100.mat", // Skip
  MAROS_MESZAROS_DIR "CONT-101.mat", // Skip
  MAROS_MESZAROS_DIR "CONT-200.mat", // Skip
  MAROS_MESZAROS_DIR "CONT-201.mat", // Skip
  MAROS_MESZAROS_DIR "CONT-300.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP1_L.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP1_M.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP1_S.mat", // ----- Pass
  MAROS_MESZAROS_DIR "CVXQP2_L.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP2_M.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP2_S.mat", // ----- Pass
  MAROS_MESZAROS_DIR "CVXQP3_L.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP3_M.mat", // Skip
  MAROS_MESZAROS_DIR "CVXQP3_S.mat", // ----- Pass
  MAROS_MESZAROS_DIR "DPKLO1.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "DTOC3.mat",    // Skip
  MAROS_MESZAROS_DIR "DUAL1.mat",    // ----- Pass
  MAROS_MESZAROS_DIR "DUAL2.mat",    // ----- Pass
  MAROS_MESZAROS_DIR "DUAL3.mat",    // ----- Pass
  MAROS_MESZAROS_DIR "DUAL4.mat",    // ----- Pass
  MAROS_MESZAROS_DIR "DUALC1.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "DUALC2.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "DUALC5.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "DUALC8.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "EXDATA.mat",   // Skip
  MAROS_MESZAROS_DIR "GENHS28.mat",  // ----- Pass
  MAROS_MESZAROS_DIR "GOULDQP2.mat", // Skip
  MAROS_MESZAROS_DIR "GOULDQP3.mat", // Skip
  MAROS_MESZAROS_DIR "HS118.mat",    // ----- Pass
  MAROS_MESZAROS_DIR "HS21.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "HS268.mat",    // ----- Pass
  MAROS_MESZAROS_DIR "HS35.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "HS35MOD.mat",  // ----- Pass
  MAROS_MESZAROS_DIR "HS51.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "HS52.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "HS53.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "HS76.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "HUES-MOD.mat", // Skip
  MAROS_MESZAROS_DIR "HUESTIS.mat",  // Skip
  MAROS_MESZAROS_DIR "KSIP.mat",     // Skip
  MAROS_MESZAROS_DIR "LASER.mat",    // Skip
  MAROS_MESZAROS_DIR "LISWET1.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET10.mat", // Skip
  MAROS_MESZAROS_DIR "LISWET11.mat", // Skip
  MAROS_MESZAROS_DIR "LISWET12.mat", // Skip
  MAROS_MESZAROS_DIR "LISWET2.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET3.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET4.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET5.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET6.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET7.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET8.mat",  // Skip
  MAROS_MESZAROS_DIR "LISWET9.mat",  // Skip
  MAROS_MESZAROS_DIR "LOTSCHD.mat",  // ----- Pass
  MAROS_MESZAROS_DIR "MOSARQP1.mat", // Skip
  MAROS_MESZAROS_DIR "MOSARQP2.mat", // Skip
  MAROS_MESZAROS_DIR "POWELL20.mat", // Skip
  MAROS_MESZAROS_DIR "PRIMAL1.mat",  // ----- Pass
  MAROS_MESZAROS_DIR "PRIMAL2.mat",  // ----- Pass
  MAROS_MESZAROS_DIR "PRIMAL3.mat",  // ----- Pass
  MAROS_MESZAROS_DIR "PRIMAL4.mat",  // Skip
  // MAROS_MESZAROS_DIR "PRIMALC1.mat", // ------------- Fail
  // MAROS_MESZAROS_DIR "PRIMALC2.mat", // ------------- Fail
  // MAROS_MESZAROS_DIR "PRIMALC5.mat", // ------------- Fail
  // MAROS_MESZAROS_DIR "PRIMALC8.mat", // ------------- Fail
  MAROS_MESZAROS_DIR "Q25FV47.mat",  // Skip
  MAROS_MESZAROS_DIR "QADLITTL.mat", // ----- Pass
  MAROS_MESZAROS_DIR "QAFIRO.mat",   // ----- Pass
  // MAROS_MESZAROS_DIR "QBANDM.mat",   // ------------- Fail
  MAROS_MESZAROS_DIR "QBEACONF.mat", // ----- Pass
  // MAROS_MESZAROS_DIR "QBORE3D.mat",  // ------------- Fail
  // MAROS_MESZAROS_DIR "QBRANDY.mat",  // ------------- Fail
  // MAROS_MESZAROS_DIR "QCAPRI.mat",   // ------------- Fail
  // MAROS_MESZAROS_DIR "QE226.mat",    // ------------- Fail
  MAROS_MESZAROS_DIR "QETAMACR.mat", // Skip
  MAROS_MESZAROS_DIR "QFFFFF80.mat", // Skip
  // MAROS_MESZAROS_DIR "QFORPLAN.mat", // ------------- Fail
  MAROS_MESZAROS_DIR "QGFRDXPN.mat", // Skip
  // MAROS_MESZAROS_DIR "QFORPLAN.mat", // ------------- Fail
  MAROS_MESZAROS_DIR "QGROW22.mat", // Skip
  // MAROS_MESZAROS_DIR "QGROW7.mat",   // ------------- Fail
  // MAROS_MESZAROS_DIR "QISRAEL.mat",  // ------------- Fail
  MAROS_MESZAROS_DIR "QPCBLEND.mat", // ----- Pass
  // MAROS_MESZAROS_DIR "QPCBOEI1.mat", // ------------- Fail
  // MAROS_MESZAROS_DIR "QPCBOEI2.mat", // ------------- Fail
  MAROS_MESZAROS_DIR "QPCSTAIR.mat", // ----- Pass
  MAROS_MESZAROS_DIR "QPILOTNO.mat", // Skip
  MAROS_MESZAROS_DIR "QPTEST.mat",   // Skip
  MAROS_MESZAROS_DIR "QRECIPE.mat",  // Skip
  MAROS_MESZAROS_DIR "QSC205.mat",   // Skip
  // MAROS_MESZAROS_DIR "QSCAGR25.mat", // ------------- Fail
  // MAROS_MESZAROS_DIR "QSCAGR7.mat",  // ------------- Fail
  // MAROS_MESZAROS_DIR "QSCFXM1.mat",  // ------------- Fail
  MAROS_MESZAROS_DIR "QSCFXM2.mat",  // Skip
  MAROS_MESZAROS_DIR "QSCFXM3.mat",  // Skip
  MAROS_MESZAROS_DIR "QSCORPIO.mat", // ----- Pass
  MAROS_MESZAROS_DIR "QSCRS8.mat",   // Skip
  MAROS_MESZAROS_DIR "QSCSD1.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "QSCSD6.mat",   // Skip
  MAROS_MESZAROS_DIR "QSCSD8.mat",   // Skip
  // MAROS_MESZAROS_DIR "QSCTAP1.mat",  // ------------- Fail
  MAROS_MESZAROS_DIR "QSCTAP2.mat", // Skip
  MAROS_MESZAROS_DIR "QSCTAP3.mat", // Skip
  MAROS_MESZAROS_DIR "QSEBA.mat",   // Skip
  // MAROS_MESZAROS_DIR "QSHARE1B.mat", // ------------- Fail
  // MAROS_MESZAROS_DIR "QSHARE2B.mat", // ------------- Fail
  MAROS_MESZAROS_DIR "QSHELL.mat",   // Skip
  MAROS_MESZAROS_DIR "QSHIP04L.mat", // Skip
  MAROS_MESZAROS_DIR "QSHIP04S.mat", // Skip
  MAROS_MESZAROS_DIR "QSHIP08L.mat", // Skip
  MAROS_MESZAROS_DIR "QSHIP08S.mat", // Skip
  MAROS_MESZAROS_DIR "QSHIP12L.mat", // Skip
  MAROS_MESZAROS_DIR "QSHIP12S.mat", // Skip
  MAROS_MESZAROS_DIR "QSIERRA.mat",  // Skip
  // MAROS_MESZAROS_DIR "QSTAIR.mat",   // ------------- Fail
  MAROS_MESZAROS_DIR "QSTANDAT.mat", // Skip
  MAROS_MESZAROS_DIR "S268.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "STADAT1.mat",  // Skip
  MAROS_MESZAROS_DIR "STADAT2.mat",  // Skip
  MAROS_MESZAROS_DIR "STADAT3.mat",  // Skip
  MAROS_MESZAROS_DIR "STCQP1.mat",   // Skip
  MAROS_MESZAROS_DIR "STCQP2.mat",   // Skip
  MAROS_MESZAROS_DIR "TAME.mat",     // ----- Pass
  MAROS_MESZAROS_DIR "UBH1.mat",     // Skip
  MAROS_MESZAROS_DIR "VALUES.mat",   // ----- Pass
  MAROS_MESZAROS_DIR "YAO.mat",      // Skip
  MAROS_MESZAROS_DIR "ZECEVIC2.mat", // ----- Pass
};

TEST_CASE("dense maros meszaros using the api")
{
  using T = double;
  using isize = dense::isize;
  Timer<T> timer;
  T elapsed_time = 0.0;

  for (auto const* file : files) {
    SUBCASE(file)
    {
      auto qp = load_qp(file);
      isize n = qp.P.rows();
      isize n_eq_in = qp.A.rows();

      const bool skip = n > 1000 || n_eq_in > 1000;
      if (skip) {
        std::cout << " path: " << qp.filename << " n: " << n
                  << " n_eq+n_in: " << n_eq_in << " - skipping" << std::endl;
      } else {
        std::cout << " path: " << qp.filename << " n: " << n
                  << " n_eq+n_in: " << n_eq_in << std::endl;
      }

      if (!skip) {

        auto preprocessed = preprocess_qp(qp);
        auto& H = preprocessed.H;
        auto& A = preprocessed.A;
        auto& C = preprocessed.C;
        auto& g = preprocessed.g;
        auto& b = preprocessed.b;
        auto& u = preprocessed.u;
        auto& l = preprocessed.l;

        isize dim = H.rows();
        isize n_eq = A.rows();
        isize n_in = C.rows();
        timer.stop();
        timer.start();
        osqp::dense::QP<T> qp{
          dim, n_eq, n_in, false, DenseBackend::PrimalDualLDLT
        }; // creating QP object
        // TODO: Automatic when PrimalDualLDLT is solved
        qp.init(H, g, A, b, C, l, u);
        qp.settings.verbose = false;

        qp.settings.eps_abs = 1e-3; // OSQP unit test
        qp.settings.eps_rel = 0;
        qp.settings.eps_primal_inf = 1e-12;
        qp.settings.eps_dual_inf = 1e-12;
        auto& eps = qp.settings.eps_abs;

        for (size_t it = 0; it < 2; ++it) {
          if (it > 0)
            qp.settings.initial_guess =
              InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;

          qp.solve();
          const auto& x = qp.results.x;
          const auto& y = qp.results.y;
          const auto& z = qp.results.z;

          T prim_eq = common::dense::infty_norm(A * x - b);
          T prim_in =
            common::dense::infty_norm(helpers::positive_part(C * x - u) +
                                      helpers::negative_part(C * x - l));
          std::cout << "primal residual " << std::max(prim_eq, prim_in)
                    << std::endl;
          std::cout << "dual residual "
                    << common::dense::infty_norm(H * x + g + A.transpose() * y +
                                                 C.transpose() * z)
                    << std::endl;
          std::cout << "iter " << qp.results.info.iter_ext << std::endl;
          CHECK(common::dense::infty_norm(H * x + g + A.transpose() * y +
                                          C.transpose() * z) < 2 * eps);
          CHECK(common::dense::infty_norm(A * x - b) > -eps);
          CHECK((C * x - l).minCoeff() > -eps);
          CHECK((C * x - u).maxCoeff() < eps);

          if (it > 0) {
            CHECK(qp.results.info.iter_ext == 0);
          }
        }
        timer.stop();
        elapsed_time += timer.elapsed().user;
      }
    }
  }
  std::cout << "timings total : \t" << elapsed_time * 1e-3 << "ms" << std::endl;
}
