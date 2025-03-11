//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <iostream>
#include <maros_meszaros.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/osqp/dense/dense.hpp>

using namespace proxsuite;

#define MAROS_MESZAROS_DIR PROBLEM_PATH "/data/maros_meszaros_data/"

char const* files[] = {
  // MAROS_MESZAROS_DIR "AUG2D.mat",    // skipped
  // MAROS_MESZAROS_DIR "AUG2DC.mat",   // skipped
  // MAROS_MESZAROS_DIR "AUG2DCQP.mat", // skipped
  // MAROS_MESZAROS_DIR "AUG2DQP.mat",  // skipped
  // MAROS_MESZAROS_DIR "AUG3D.mat",    // skipped
  // MAROS_MESZAROS_DIR "AUG3DC.mat",   // skipped
  // MAROS_MESZAROS_DIR "AUG3DCQP.mat", // skipped
  // MAROS_MESZAROS_DIR "AUG3DQP.mat",  // skipped
  // MAROS_MESZAROS_DIR "BOYD1.mat",    // skipped
  // MAROS_MESZAROS_DIR "BOYD2.mat",    // skipped
  // MAROS_MESZAROS_DIR "CONT-050.mat", // skipped
  // MAROS_MESZAROS_DIR "CONT-100.mat", // skipped
  // MAROS_MESZAROS_DIR "CONT-101.mat", // skipped
  // MAROS_MESZAROS_DIR "CONT-200.mat", // skipped
  // MAROS_MESZAROS_DIR "CONT-201.mat", // skipped
  // MAROS_MESZAROS_DIR "CONT-300.mat", // skipped
  // MAROS_MESZAROS_DIR "CVXQP1_L.mat", // skipped
  // MAROS_MESZAROS_DIR "CVXQP1_M.mat", // skipped
  MAROS_MESZAROS_DIR
  "CVXQP1_S.mat", // failed - 1st r_prim slightly conv then plateau, r_dual and
                  // r_g plateau, then polish makes worst - 2nd all diverges
  // MAROS_MESZAROS_DIR "CVXQP2_L.mat", // skipped
  // MAROS_MESZAROS_DIR "CVXQP2_M.mat", // skipped
  // MAROS_MESZAROS_DIR "CVXQP2_S.mat", // failed - 1st r_prim and r_dual conv,
  // r_g plateau, then polish make r_prim/dual explode - 2nd all diverges
  // MAROS_MESZAROS_DIR "CVXQP3_L.mat", // skipped
  // MAROS_MESZAROS_DIR "CVXQP3_M.mat", // skipped
  // MAROS_MESZAROS_DIR "CVXQP3_S.mat", // failed - 1st r_prim and r_dual conv,
  // r_g plateau, then polish make r_prim/dual explode - 2nd all diverges
  // MAROS_MESZAROS_DIR "DPKLO1.mat",   // passed
  // MAROS_MESZAROS_DIR "DTOC3.mat",    // skipped
  // MAROS_MESZAROS_DIR "DUAL1.mat",    // passed
  // MAROS_MESZAROS_DIR "DUAL2.mat",    // passed
  // MAROS_MESZAROS_DIR "DUAL3.mat",    // passed
  // MAROS_MESZAROS_DIR "DUAL4.mat",    // passed
  // MAROS_MESZAROS_DIR "DUALC1.mat",   // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "DUALC2.mat",   // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "DUALC5.mat",   // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "DUALC8.mat",   // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "EXDATA.mat",   // skipped
  // MAROS_MESZAROS_DIR "GENHS28.mat",  // compile error at "checkout" in polish
  // MAROS_MESZAROS_DIR "GOULDQP2.mat", // skipped
  // MAROS_MESZAROS_DIR "GOULDQP3.mat", // skipped
  // MAROS_MESZAROS_DIR "HS118.mat",    // passed
  // MAROS_MESZAROS_DIR "HS21.mat",     // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "HS268.mat",    // failed, r_prim conv well, but r_dual
  // and r_g plateau and max iter reached
  // MAROS_MESZAROS_DIR "HS35.mat",     // failed, converges slowly, polish made
  // it worst
  // MAROS_MESZAROS_DIR "HS35MOD.mat",  // compile error at "checkout" in polish
  // MAROS_MESZAROS_DIR "HS51.mat",     // compile error at "checkout" in polish
  // MAROS_MESZAROS_DIR "HS52.mat",     // passed
  // MAROS_MESZAROS_DIR "HS53.mat",     // passed
  // MAROS_MESZAROS_DIR "HS76.mat",     // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "HUES-MOD.mat", // skipped
  // MAROS_MESZAROS_DIR "HUESTIS.mat",  // skipped
  // MAROS_MESZAROS_DIR "KSIP.mat",     // skipped
  // MAROS_MESZAROS_DIR "LASER.mat",    // skipped
  // MAROS_MESZAROS_DIR "LISWET1.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET10.mat", // skipped
  // MAROS_MESZAROS_DIR "LISWET11.mat", // skipped
  // MAROS_MESZAROS_DIR "LISWET12.mat", // skipped
  // MAROS_MESZAROS_DIR "LISWET2.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET3.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET4.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET5.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET6.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET7.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET8.mat",  // skipped
  // MAROS_MESZAROS_DIR "LISWET9.mat",  // skipped
  // MAROS_MESZAROS_DIR "LOTSCHD.mat",  // passed
  // MAROS_MESZAROS_DIR "MOSARQP1.mat", // skipped
  // MAROS_MESZAROS_DIR "MOSARQP2.mat", // skipped
  // MAROS_MESZAROS_DIR "POWELL20.mat", // skipped
  // MAROS_MESZAROS_DIR "PRIMAL1.mat",  // failed, seemed to converge well,
  // polish made it worst
  // MAROS_MESZAROS_DIR "PRIMAL2.mat",
  // MAROS_MESZAROS_DIR "PRIMAL3.mat",
  // MAROS_MESZAROS_DIR "PRIMAL4.mat",
  // MAROS_MESZAROS_DIR "PRIMALC1.mat",
  // MAROS_MESZAROS_DIR "PRIMALC2.mat",
  // MAROS_MESZAROS_DIR "PRIMALC5.mat",
  // MAROS_MESZAROS_DIR "PRIMALC8.mat",
  // MAROS_MESZAROS_DIR "Q25FV47.mat",
  // MAROS_MESZAROS_DIR "QADLITTL.mat",
  // MAROS_MESZAROS_DIR "QAFIRO.mat",
  // MAROS_MESZAROS_DIR "QBANDM.mat",
  // MAROS_MESZAROS_DIR "QBEACONF.mat",
  // MAROS_MESZAROS_DIR "QBORE3D.mat",
  // MAROS_MESZAROS_DIR "QBRANDY.mat",
  // MAROS_MESZAROS_DIR "QCAPRI.mat",
  // MAROS_MESZAROS_DIR "QE226.mat",
  // MAROS_MESZAROS_DIR "QETAMACR.mat",
  // MAROS_MESZAROS_DIR "QFFFFF80.mat",
  // MAROS_MESZAROS_DIR "QFORPLAN.mat",
  // MAROS_MESZAROS_DIR "QGFRDXPN.mat",
  // MAROS_MESZAROS_DIR "QGROW15.mat",
  // MAROS_MESZAROS_DIR "QGROW22.mat",
  // MAROS_MESZAROS_DIR "QGROW7.mat",
  // MAROS_MESZAROS_DIR "QISRAEL.mat",
  // MAROS_MESZAROS_DIR "QPCBLEND.mat",
  // MAROS_MESZAROS_DIR "QPCBOEI1.mat",
  // MAROS_MESZAROS_DIR "QPCBOEI2.mat",
  // MAROS_MESZAROS_DIR "QPCSTAIR.mat",
  // MAROS_MESZAROS_DIR "QPILOTNO.mat",
  // MAROS_MESZAROS_DIR "QPTEST.mat",
  // MAROS_MESZAROS_DIR "QRECIPE.mat",
  // MAROS_MESZAROS_DIR "QSC205.mat",
  // MAROS_MESZAROS_DIR "QSCAGR25.mat",
  // MAROS_MESZAROS_DIR "QSCAGR7.mat",
  // MAROS_MESZAROS_DIR "QSCFXM1.mat",
  // MAROS_MESZAROS_DIR "QSCFXM2.mat",
  // MAROS_MESZAROS_DIR "QSCFXM3.mat",
  // MAROS_MESZAROS_DIR "QSCORPIO.mat",
  // MAROS_MESZAROS_DIR "QSCRS8.mat",
  // MAROS_MESZAROS_DIR "QSCSD1.mat",
  // MAROS_MESZAROS_DIR "QSCSD6.mat",
  // MAROS_MESZAROS_DIR "QSCSD8.mat",
  // MAROS_MESZAROS_DIR "QSCTAP1.mat",
  // MAROS_MESZAROS_DIR "QSCTAP2.mat",
  // MAROS_MESZAROS_DIR "QSCTAP3.mat",
  // MAROS_MESZAROS_DIR "QSEBA.mat",
  // MAROS_MESZAROS_DIR "QSHARE1B.mat",
  // MAROS_MESZAROS_DIR "QSHARE2B.mat",
  // MAROS_MESZAROS_DIR "QSHELL.mat",
  // MAROS_MESZAROS_DIR "QSHIP04L.mat",
  // MAROS_MESZAROS_DIR "QSHIP04S.mat",
  // MAROS_MESZAROS_DIR "QSHIP08L.mat",
  // MAROS_MESZAROS_DIR "QSHIP08S.mat",
  // MAROS_MESZAROS_DIR "QSHIP12L.mat",
  // MAROS_MESZAROS_DIR "QSHIP12S.mat",
  // MAROS_MESZAROS_DIR "QSIERRA.mat",
  // MAROS_MESZAROS_DIR "QSTAIR.mat",
  // MAROS_MESZAROS_DIR "QSTANDAT.mat",
  // MAROS_MESZAROS_DIR "S268.mat",
  // MAROS_MESZAROS_DIR "STADAT1.mat",
  // MAROS_MESZAROS_DIR "STADAT2.mat",
  // MAROS_MESZAROS_DIR "STADAT3.mat",
  // MAROS_MESZAROS_DIR "STCQP1.mat",
  // MAROS_MESZAROS_DIR "STCQP2.mat",
  // MAROS_MESZAROS_DIR "TAME.mat",     // compile error here at "checkout" in
  // polish
  // MAROS_MESZAROS_DIR "UBH1.mat",
  // MAROS_MESZAROS_DIR "VALUES.mat",
  // MAROS_MESZAROS_DIR "YAO.mat",
  // MAROS_MESZAROS_DIR "ZECEVIC2.mat",
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
      osqp::dense::QP<T> qp{
        dim, n_eq, n_in, false, proxsuite::proxqp::DenseBackend::PrimalDualLDLT
      }; // creating QP object
      qp.init(H, g, A, b, C, l, u);

      // qp.settings.eps_abs = 2e-8;
      // auto& eps = qp.settings.eps_abs;
      // qp.settings.eps_abs = 1e-3;
      // qp.settings.eps_rel = 0;
      qp.settings.eps_abs = 2e-12;
      auto& eps = qp.settings.eps_abs;
      qp.settings.eps_abs = 2e-12;
      qp.settings.eps_rel = 0;
      // qp.settings.eps_primal_inf = 1e-12;
      // qp.settings.eps_dual_inf = 1e-12;
      qp.settings.default_mu_eq = 1.e-2;
      qp.settings.default_mu_in = 1.e1;
      // qp.settings.max_iter = 10000;
      qp.settings.max_iter = 50000;
      qp.settings.verbose = true;

      // for (size_t it = 0; it < 2; ++it) {
      for (size_t it = 0; it < 1; ++it) {
        if (it > 0)
          qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::
            WARM_START_WITH_PREVIOUS_RESULT;

        qp.solve();
        const auto& x = qp.results.x;
        const auto& y = qp.results.y;
        const auto& z = qp.results.z;
        // std::cout << x << std::endl;
        // std::cout << y << std::endl;
        // std::cout << z << std::endl;

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
        std::cout << "iter " << qp.results.info.iter_ext << std::endl;
        CHECK(proxqp::dense::infty_norm(H * x + g + A.transpose() * y +
                                        C.transpose() * z) < 2 * eps);
        CHECK(proxqp::dense::infty_norm(A * x - b) > -eps);
        CHECK((C * x - l).minCoeff() > -eps);
        CHECK((C * x - u).maxCoeff() < eps);

        qp.settings.verbose = false;
      }

      timer.stop();
      elapsed_time += timer.elapsed().user;
    }
  }
  std::cout << "timings total : \t" << elapsed_time * 1e-3 << "ms" << std::endl;
}

// Settings test:
// PrimalDualLDLT
// No mu_update

// Note test:
// It does not pass

// Problems "HS51.mat" and "TAME.mat" -> same error as in osqp_cvxpy, ie:
// malloc(): invalid size (unsorted) / malloc(): unaligned tcache chunk detected

// Work

// Question: Why we consider some problems, but they are automatically skipped ?

// Note, often when osqp fails (diverges), warm starting is even worst (when I
// just put 10 iters)

// Note, the failed, osqp diverge comment can also denote just staying

// Note, when polishing step make r_prim/dual worst, it is when r_g does not
// converge too

// Note, maths guarantees of mu_update to help to converge, or should converge
// in any case ?

// Todo: Make the polishing more robust / efficient to make the tests pass