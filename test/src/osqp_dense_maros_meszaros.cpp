//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <maros_meszaros.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/osqp/dense/dense.hpp>

using namespace proxsuite;
namespace pp = proxsuite::proxqp;
namespace pod = proxsuite::osqp::dense;

#define MAROS_MESZAROS_DIR PROBLEM_PATH "/data/maros_meszaros_data/"

char const* files[] = {
  //   MAROS_MESZAROS_DIR "AUG2D.mat",              // skipping
  //   MAROS_MESZAROS_DIR "AUG2DC.mat",             // skipping
  //   MAROS_MESZAROS_DIR "AUG2DCQP.mat",           // skipping
  //   MAROS_MESZAROS_DIR "AUG2DQP.mat",            // skipping
  //   MAROS_MESZAROS_DIR "AUG3D.mat",              // skipping
  //   MAROS_MESZAROS_DIR "AUG3DC.mat",             // skipping
  //   MAROS_MESZAROS_DIR "AUG3DCQP.mat",           // skipping
  //   MAROS_MESZAROS_DIR "AUG3DQP.mat",            // skipping
  //   MAROS_MESZAROS_DIR "BOYD1.mat",              // skipping
  //   MAROS_MESZAROS_DIR "BOYD2.mat",              // skipping
  //   MAROS_MESZAROS_DIR "CONT-050.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CONT-100.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CONT-101.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CONT-200.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CONT-201.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CONT-300.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP1_L.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP1_M.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP1_S.mat",  // Pass
  //   MAROS_MESZAROS_DIR "CVXQP2_L.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP2_M.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP2_S.mat",  // Pass
  //   MAROS_MESZAROS_DIR "CVXQP3_L.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP3_M.mat",           // skipping
  //   MAROS_MESZAROS_DIR "CVXQP3_S.mat",  // Pass
  //   MAROS_MESZAROS_DIR "DPKLO1.mat",    // Pass
  //   MAROS_MESZAROS_DIR "DTOC3.mat",              // Skipping
  //   MAROS_MESZAROS_DIR "DUAL1.mat",     // Pass
  //   MAROS_MESZAROS_DIR "DUAL2.mat",     // Pass
  //   MAROS_MESZAROS_DIR "DUAL3.mat",     // Pass
  //   MAROS_MESZAROS_DIR "DUAL4.mat",     // Pass
  //   MAROS_MESZAROS_DIR "DUALC1.mat",    // Pass
  //   MAROS_MESZAROS_DIR "DUALC2.mat",    // Pass
  //   MAROS_MESZAROS_DIR "DUALC5.mat",    // Pass
  //   MAROS_MESZAROS_DIR "DUALC8.mat",    // Pass
  //   MAROS_MESZAROS_DIR "EXDATA.mat",             // Skipping
  //   MAROS_MESZAROS_DIR "GENHS28.mat",   // Pass
  //   MAROS_MESZAROS_DIR "GOULDQP2.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "GOULDQP3.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "HS118.mat",     // Pass
  //   MAROS_MESZAROS_DIR "HS21.mat",      // Pass
  //   MAROS_MESZAROS_DIR "HS268.mat",     // Pass
  //   MAROS_MESZAROS_DIR "HS35.mat",      // Pass
  //   MAROS_MESZAROS_DIR "HS35MOD.mat",   // Pass
  //   MAROS_MESZAROS_DIR "HS51.mat",      // Pass
  //   MAROS_MESZAROS_DIR "HS52.mat",      // Pass
  //   MAROS_MESZAROS_DIR "HS53.mat",      // Pass
  //   MAROS_MESZAROS_DIR "HS76.mat",      // Pass
  //   MAROS_MESZAROS_DIR "HUES-MOD.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "HUESTIS.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "KSIP.mat",               // Skipping
  //   MAROS_MESZAROS_DIR "LASER.mat",              // Skipping
  //   MAROS_MESZAROS_DIR "LISWET1.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET10.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "LISWET11.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "LISWET12.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "LISWET2.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET3.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET4.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET5.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET6.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET7.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET8.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LISWET9.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "LOTSCHD.mat",   // Pass
  //   MAROS_MESZAROS_DIR "MOSARQP1.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "MOSARQP2.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "POWELL20.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "PRIMAL1.mat",   // Pass
  //   MAROS_MESZAROS_DIR "PRIMAL2.mat",   // Pass
  //   MAROS_MESZAROS_DIR "PRIMAL3.mat",   // Pass
  //   MAROS_MESZAROS_DIR "PRIMAL4.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "PRIMALC1.mat",                          // Fail:
  //   Values nan (primal residual 0, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 11) n: 230 n_eq+n_in: 239
  //   MAROS_MESZAROS_DIR "PRIMALC2.mat",                          // Fail:
  //   Values nan (primal residual 0, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 8) n: 231 n_eq+n_in: 238
  //   MAROS_MESZAROS_DIR "PRIMALC5.mat",                          // Fail:
  //   Values nan (primal residual 0, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 9) n: 287 n_eq+n_in: 295
  //   MAROS_MESZAROS_DIR "PRIMALC8.mat",                          // Fail:
  //   Values nan (primal residual 0, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 8) n: 520 n_eq+n_in: 528
  //   MAROS_MESZAROS_DIR "Q25FV47.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "QADLITTL.mat",                          // Fail:
  //   Values nan (primal residual huge, dual residual nan, iter 10000, polish
  //   2, polish failed, mu updates 4) n: 97 n_eq+n_in: 153
  //   MAROS_MESZAROS_DIR "QAFIRO.mat",                            // Fail:
  //   Values nan (primal residual nan, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 3) n: 32 n_eq+n_in: 59
  //   MAROS_MESZAROS_DIR "QBANDM.mat",     // Pass
  //   MAROS_MESZAROS_DIR "QBEACONF.mat",   // Pass
  //   MAROS_MESZAROS_DIR "QBORE3D.mat",                           // Fail: Does
  //   not converge enough (primal residual 1e-1, dual residual 8e-3, iter
  //   10000, polish not run, mu updates 1) n: 315 n_eq+n_in: 548
  //   MAROS_MESZAROS_DIR "QBRANDY.mat",                           // Fail: Does
  //   not converge enough (primal residual 7e-2, dual residual 4e-2, iter
  //   10000, polish not run, mu updates 2) n: 249 n_eq+n_in: 469
  //   MAROS_MESZAROS_DIR "QCAPRI.mat",                            // Fail: Does
  //   not converge well (primal residual 7e-3, dual residual 7e-1,  iter 10000,
  //   polish not run, mu updates 12) n: 353 n_eq+n_in: 624
  //   MAROS_MESZAROS_DIR "QE226.mat",                             // Fail: Does
  //   not converge enough (primal residual 2e-3, dual residual 2e-3, iter
  //   10000, polish not run, mu updates 1) n: 282 n_eq+n_in: 505
  //   MAROS_MESZAROS_DIR "QETAMACR.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QFFFFF80.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QFORPLAN.mat",                          // Fail: Does
  //   not converge enough (primal residual 1e-2, dual residual 1e-1, iter
  //   10000, polish not run, mu updates 10) n: 421 n_eq+n_in: 582
  //   MAROS_MESZAROS_DIR "QGFRDXPN.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QGROW15.mat",                           // Fail: Does
  //   not converge well (primal residual 6e3, dual residual 2e-2,  iter 10000,
  //   polish not run, mu updates 50) n: 645 n_eq+n_in: 945
  //   MAROS_MESZAROS_DIR "QGROW22.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "QGROW7.mat",                            // Fail: Does
  //   not converge well (primal residual 8e3, dual residual 6e-1,  iter 10000,
  //   polish not run, mu updates 64) n: 301 n_eq+n_in: 441
  //   MAROS_MESZAROS_DIR "QISRAEL.mat",                           // Fail: Does
  //   not converge well (primal residual 4e2, dual residual 1e3,   iter 10000,
  //   polish not run, mu updates 30) n: 142 n_eq+n_in: 316
  //   MAROS_MESZAROS_DIR "QPCBLEND.mat",                          // Fail:
  //   Values nan (primal residual nan, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 3) n: 83 n_eq+n_in: 157
  //   MAROS_MESZAROS_DIR "QPCBOEI1.mat",                          // Fail: Does
  //   not converge enough (primal residual 1e-1, dual residual 1e-3, iter
  //   10000, polish not run, mu updates 1) n: 384 n_eq+n_in: 735
  //   MAROS_MESZAROS_DIR "QPCBOEI2.mat",                          // Fail: Does
  //   not converge well (primal residual 1e-2, dual residual 1e1,  iter 10000,
  //   polish not run, mu updates 4) n: 143 n_eq+n_in: 309
  //   MAROS_MESZAROS_DIR "QPCSTAIR.mat",  // Pass
  //   MAROS_MESZAROS_DIR "QPILOTNO.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QPTEST.mat",    // Pass
  //   MAROS_MESZAROS_DIR "QRECIPE.mat",   // Pass
  //   MAROS_MESZAROS_DIR "QSC205.mat",                            // Fail:
  //   Values nan (primal residual nan, dual residual nan, iter 10000, polish 2,
  //   polish failed, mu updates 4) n: 203 n_eq+n_in: 408
  //   MAROS_MESZAROS_DIR "QSCAGR25.mat",                          // Fail: Does
  //   not converge enough (primal residual 6e-2, dual residual 6e-2, iter
  //   10000, polish not run, mu updates 3) n: 500 n_eq+n_in: 971
  //   MAROS_MESZAROS_DIR "QSCAGR7.mat",                           // Fail: Does
  //   not converge enough (primal residual 9e-2, dual residual 1e0, iter 10000,
  //   polish not run, mu updates 12) n: 140 n_eq+n_in: 269
  //   MAROS_MESZAROS_DIR "QSCFXM1.mat",                           // Fail: Does
  //   not converge enough (primal residual 3e-2, dual residual 4e-2, iter
  //   10000, polish not run, mu updates 1) n: 457 n_eq+n_in: 787
  //   MAROS_MESZAROS_DIR "QSCFXM2.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "QSCFXM3.mat",            // Skipping
  MAROS_MESZAROS_DIR
  "QSCORPIO.mat", // Fail: Does not converge well and polish -> ?? (primal
                  // residual 2e-4, dual residual 7e-1, iter 10000, polish 2,
                  // polish failed, mu updates 18) n: 358 n_eq+n_in: 746
  //   MAROS_MESZAROS_DIR "QSCRS8.mat",             // Skipping
  MAROS_MESZAROS_DIR
  "QSCSD1.mat", // Fail: Does not converge well and polish -> ?? (primal
                // residual 1e-2, dual residual 1e-2, iter 10000, polish 2,
                // polish failed, mu updates 134) n: 760 n_eq+n_in: 837
  //   MAROS_MESZAROS_DIR "QSCSD6.mat",             // Skipping
  //   MAROS_MESZAROS_DIR "QSCSD8.mat",             // Skipping
  //   MAROS_MESZAROS_DIR "QSCTAP1.mat",                          // Fail: Does
  //   not converge well (primal residual 5e-3, dual residual 1e0, iter 10000,
  //   polish not run, mu updates 16) n: 480 n_eq+n_in: 780
  //   MAROS_MESZAROS_DIR "QSCTAP2.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "QSCTAP3.mat",
  //   MAROS_MESZAROS_DIR "QSEBA.mat",              // Skipping
  //   MAROS_MESZAROS_DIR "QSHARE1B.mat",                         // Fail: Does
  //   not converge well (primal residual 3e0, dual residual 3e-3, iter 10000,
  //   polish not run, mu updates 4) n: 225 n_eq+n_in: 342
  //   MAROS_MESZAROS_DIR "QSHARE2B.mat",                         // Fail: Does
  //   not converge enough (primal residual 1e-1, dual residual 8e-1, iter
  //   10000, polish not run, mu updates 4) n: 79 n_eq+n_in: 175
  //   MAROS_MESZAROS_DIR "QSHELL.mat",             // Skipping
  //   MAROS_MESZAROS_DIR "QSHIP04L.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QSHIP04S.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QSHIP08L.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QSHIP08S.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QSHIP12L.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QSHIP12S.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "QSIERRA.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "QSTAIR.mat",                          // Fail: Does
  //   not converge enough (primal residual 2e-2, dual residual 3e-1, iter
  //   10000, polish not run, mu updates 13) n: 467 n_eq+n_in: 823
  //   MAROS_MESZAROS_DIR "QSTANDAT.mat",           // Skipping
  //   MAROS_MESZAROS_DIR "S268.mat",      // Pass
  //   MAROS_MESZAROS_DIR "STADAT1.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "STADAT2.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "STADAT3.mat",            // Skipping
  //   MAROS_MESZAROS_DIR "STCQP1.mat",             // Skipping
  //   MAROS_MESZAROS_DIR "STCQP2.mat",             // Skipping
  //   MAROS_MESZAROS_DIR "TAME.mat",      // Pass
  //   MAROS_MESZAROS_DIR "UBH1.mat",               // Skipping
  //   MAROS_MESZAROS_DIR "VALUES.mat",    // Pass
  //   MAROS_MESZAROS_DIR "YAO.mat",                // Skipping
  //   MAROS_MESZAROS_DIR "ZECEVIC2.mat",  // Pass
};

TEST_CASE("dense maros meszaros using the api")
{
  using T = double;
  using isize = proxqp::utils::isize;
  proxsuite::proxqp::Timer<T> timer;
  T elapsed_time = 0.0;

  for (auto const* file : files) {
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
      pod::QP<T> qp{
        dim, n_eq, n_in, false, proxsuite::proxqp::DenseBackend::PrimalDualLDLT
      }; // creating QP object

      qp.settings.default_mu_eq = T(1.E-2);
      qp.settings.default_mu_in = T(1.E1);

      qp.init(H, g, A, b, C, l, u);

      qp.settings.eps_abs = 2e-8;
      qp.settings.eps_rel = 0;
      qp.settings.eps_primal_inf = 1e-12;
      qp.settings.eps_dual_inf = 1e-12;

      {
        qp.settings.max_iter = 10000;
        qp.settings.verbose = false;
      }

      auto& eps = qp.settings.eps_abs;

      //   for (size_t it = 0; it < 2; ++it) {
      size_t it = 0; // Just test first solve for a first glance
      {
        if (it > 0)
          qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::
            WARM_START_WITH_PREVIOUS_RESULT;

        qp.solve();
        const auto& x = qp.results.x;
        const auto& y = qp.results.y;
        const auto& z = qp.results.z;

        T prim_eq = proxqp::dense::infty_norm(A * x - b);
        T prim_in =
          proxqp::dense::infty_norm(helpers::positive_part(C * x - u) +
                                    helpers::negative_part(C * x - l));
        std::cout << "primal residual " << std::max(prim_eq, prim_in)
                  << std::endl;
        std::cout << "dual residual "
                  << proxqp::dense::infty_norm(H * x + g + A.transpose() * y +
                                               C.transpose() * z)
                  << std::endl;
        std::cout << "admm iter    " << qp.results.info.iter_ext << std::endl;
        {
          std::cout << "polish calls " << qp.results.info.polish_calls
                    << std::endl;
          switch (qp.results.info.polish_status) {
            case pp::PolishStatus::POLISH_NOT_RUN: {
              std::cout << "polish not run" << std::endl;
              break;
            }
            case pp::PolishStatus::POLISH_SUCCEED: {
              std::cout << "polish succeeded" << std::endl;
              break;
            }
            case pp::PolishStatus::POLISH_FAILED: {
              std::cout << "polish failed" << std::endl;
              break;
            }
            case pp::PolishStatus::POLISH_NO_ACTIVE_SET_FOUND: {
              std::cout << "polish no active set found" << std::endl;
              break;
            }
          }
          std::cout << "mu updates " << qp.results.info.mu_updates << std::endl;
        }
        CHECK(proxqp::dense::infty_norm(H * x + g + A.transpose() * y +
                                        C.transpose() * z) < 2 * eps);
        CHECK(proxqp::dense::infty_norm(A * x - b) > -eps);
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
  std::cout << "timings total : \t" << elapsed_time * 1e-3 << "ms" << std::endl;
}
