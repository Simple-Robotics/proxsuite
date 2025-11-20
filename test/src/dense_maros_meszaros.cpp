//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <maros_meszaros.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

using namespace proxsuite;

#define MAROS_MESZAROS_DIR PROBLEM_PATH "/data/maros_meszaros_data/"

char const* files[] = {
  MAROS_MESZAROS_DIR "AUG2D.mat",    MAROS_MESZAROS_DIR "AUG2DC.mat",
  MAROS_MESZAROS_DIR "AUG2DCQP.mat", MAROS_MESZAROS_DIR "AUG2DQP.mat",
  MAROS_MESZAROS_DIR "AUG3D.mat",    MAROS_MESZAROS_DIR "AUG3DC.mat",
  MAROS_MESZAROS_DIR "AUG3DCQP.mat", MAROS_MESZAROS_DIR "AUG3DQP.mat",
  MAROS_MESZAROS_DIR "BOYD1.mat",    MAROS_MESZAROS_DIR "BOYD2.mat",
  MAROS_MESZAROS_DIR "CONT-050.mat", MAROS_MESZAROS_DIR "CONT-100.mat",
  MAROS_MESZAROS_DIR "CONT-101.mat", MAROS_MESZAROS_DIR "CONT-200.mat",
  MAROS_MESZAROS_DIR "CONT-201.mat", MAROS_MESZAROS_DIR "CONT-300.mat",
  MAROS_MESZAROS_DIR "CVXQP1_L.mat", MAROS_MESZAROS_DIR "CVXQP1_M.mat",
  MAROS_MESZAROS_DIR "CVXQP1_S.mat", MAROS_MESZAROS_DIR "CVXQP2_L.mat",
  MAROS_MESZAROS_DIR "CVXQP2_M.mat", MAROS_MESZAROS_DIR "CVXQP2_S.mat",
  MAROS_MESZAROS_DIR "CVXQP3_L.mat", MAROS_MESZAROS_DIR "CVXQP3_M.mat",
  MAROS_MESZAROS_DIR "CVXQP3_S.mat", MAROS_MESZAROS_DIR "DPKLO1.mat",
  MAROS_MESZAROS_DIR "DTOC3.mat",    MAROS_MESZAROS_DIR "DUAL1.mat",
  MAROS_MESZAROS_DIR "DUAL2.mat",    MAROS_MESZAROS_DIR "DUAL3.mat",
  MAROS_MESZAROS_DIR "DUAL4.mat",    MAROS_MESZAROS_DIR "DUALC1.mat",
  MAROS_MESZAROS_DIR "DUALC2.mat",   MAROS_MESZAROS_DIR "DUALC5.mat",
  MAROS_MESZAROS_DIR "DUALC8.mat",   MAROS_MESZAROS_DIR "EXDATA.mat",
  MAROS_MESZAROS_DIR "GENHS28.mat",  MAROS_MESZAROS_DIR "GOULDQP2.mat",
  MAROS_MESZAROS_DIR "GOULDQP3.mat", MAROS_MESZAROS_DIR "HS118.mat",
  MAROS_MESZAROS_DIR "HS21.mat",     MAROS_MESZAROS_DIR "HS268.mat",
  MAROS_MESZAROS_DIR "HS35.mat",     MAROS_MESZAROS_DIR "HS35MOD.mat",
  MAROS_MESZAROS_DIR "HS51.mat",     MAROS_MESZAROS_DIR "HS52.mat",
  MAROS_MESZAROS_DIR "HS53.mat",     MAROS_MESZAROS_DIR "HS76.mat",
  MAROS_MESZAROS_DIR "HUES-MOD.mat", MAROS_MESZAROS_DIR "HUESTIS.mat",
  MAROS_MESZAROS_DIR "KSIP.mat",     MAROS_MESZAROS_DIR "LASER.mat",
  MAROS_MESZAROS_DIR "LISWET1.mat",  MAROS_MESZAROS_DIR "LISWET10.mat",
  MAROS_MESZAROS_DIR "LISWET11.mat", MAROS_MESZAROS_DIR "LISWET12.mat",
  MAROS_MESZAROS_DIR "LISWET2.mat",  MAROS_MESZAROS_DIR "LISWET3.mat",
  MAROS_MESZAROS_DIR "LISWET4.mat",  MAROS_MESZAROS_DIR "LISWET5.mat",
  MAROS_MESZAROS_DIR "LISWET6.mat",  MAROS_MESZAROS_DIR "LISWET7.mat",
  MAROS_MESZAROS_DIR "LISWET8.mat",  MAROS_MESZAROS_DIR "LISWET9.mat",
  MAROS_MESZAROS_DIR "LOTSCHD.mat",  MAROS_MESZAROS_DIR "MOSARQP1.mat",
  MAROS_MESZAROS_DIR "MOSARQP2.mat", MAROS_MESZAROS_DIR "POWELL20.mat",
  MAROS_MESZAROS_DIR "PRIMAL1.mat",  MAROS_MESZAROS_DIR "PRIMAL2.mat",
  MAROS_MESZAROS_DIR "PRIMAL3.mat",  MAROS_MESZAROS_DIR "PRIMAL4.mat",
  MAROS_MESZAROS_DIR "PRIMALC1.mat", MAROS_MESZAROS_DIR "PRIMALC2.mat",
  MAROS_MESZAROS_DIR "PRIMALC5.mat", MAROS_MESZAROS_DIR "PRIMALC8.mat",
  MAROS_MESZAROS_DIR "Q25FV47.mat",  MAROS_MESZAROS_DIR "QADLITTL.mat",
  MAROS_MESZAROS_DIR "QAFIRO.mat",   MAROS_MESZAROS_DIR "QBANDM.mat",
  MAROS_MESZAROS_DIR "QBEACONF.mat", MAROS_MESZAROS_DIR "QBORE3D.mat",
  MAROS_MESZAROS_DIR "QBRANDY.mat",  MAROS_MESZAROS_DIR "QCAPRI.mat",
  MAROS_MESZAROS_DIR "QE226.mat",    MAROS_MESZAROS_DIR "QETAMACR.mat",
  MAROS_MESZAROS_DIR "QFFFFF80.mat", MAROS_MESZAROS_DIR "QFORPLAN.mat",
  MAROS_MESZAROS_DIR "QGFRDXPN.mat", MAROS_MESZAROS_DIR "QGROW15.mat",
  MAROS_MESZAROS_DIR "QGROW22.mat",  MAROS_MESZAROS_DIR "QGROW7.mat",
  MAROS_MESZAROS_DIR "QISRAEL.mat",  MAROS_MESZAROS_DIR "QPCBLEND.mat",
  MAROS_MESZAROS_DIR "QPCBOEI1.mat", MAROS_MESZAROS_DIR "QPCBOEI2.mat",
  MAROS_MESZAROS_DIR "QPCSTAIR.mat", MAROS_MESZAROS_DIR "QPILOTNO.mat",
  MAROS_MESZAROS_DIR "QPTEST.mat",   MAROS_MESZAROS_DIR "QRECIPE.mat",
  MAROS_MESZAROS_DIR "QSC205.mat",   MAROS_MESZAROS_DIR "QSCAGR25.mat",
  MAROS_MESZAROS_DIR "QSCAGR7.mat",  MAROS_MESZAROS_DIR "QSCFXM1.mat",
  MAROS_MESZAROS_DIR "QSCFXM2.mat",  MAROS_MESZAROS_DIR "QSCFXM3.mat",
  MAROS_MESZAROS_DIR "QSCORPIO.mat", MAROS_MESZAROS_DIR "QSCRS8.mat",
  MAROS_MESZAROS_DIR "QSCSD1.mat",   MAROS_MESZAROS_DIR "QSCSD6.mat",
  MAROS_MESZAROS_DIR "QSCSD8.mat",   MAROS_MESZAROS_DIR "QSCTAP1.mat",
  MAROS_MESZAROS_DIR "QSCTAP2.mat",  MAROS_MESZAROS_DIR "QSCTAP3.mat",
  MAROS_MESZAROS_DIR "QSEBA.mat",    MAROS_MESZAROS_DIR "QSHARE1B.mat",
  MAROS_MESZAROS_DIR "QSHARE2B.mat", MAROS_MESZAROS_DIR "QSHELL.mat",
  MAROS_MESZAROS_DIR "QSHIP04L.mat", MAROS_MESZAROS_DIR "QSHIP04S.mat",
  MAROS_MESZAROS_DIR "QSHIP08L.mat", MAROS_MESZAROS_DIR "QSHIP08S.mat",
  MAROS_MESZAROS_DIR "QSHIP12L.mat", MAROS_MESZAROS_DIR "QSHIP12S.mat",
  MAROS_MESZAROS_DIR "QSIERRA.mat",  MAROS_MESZAROS_DIR "QSTAIR.mat",
  MAROS_MESZAROS_DIR "QSTANDAT.mat", MAROS_MESZAROS_DIR "S268.mat",
  MAROS_MESZAROS_DIR "STADAT1.mat",  MAROS_MESZAROS_DIR "STADAT2.mat",
  MAROS_MESZAROS_DIR "STADAT3.mat",  MAROS_MESZAROS_DIR "STCQP1.mat",
  MAROS_MESZAROS_DIR "STCQP2.mat",   MAROS_MESZAROS_DIR "TAME.mat",
  MAROS_MESZAROS_DIR "UBH1.mat",     MAROS_MESZAROS_DIR "VALUES.mat",
  MAROS_MESZAROS_DIR "YAO.mat",      MAROS_MESZAROS_DIR "ZECEVIC2.mat",
};

TEST_CASE("dense maros meszaros using the api")
{
  using T = double;
  using isize = proxqp::utils::isize;
  proxsuite::proxqp::Timer<T> timer;
  T elapsed_time = 0.0;

  for (auto const* file : files) {
    auto qp = load_qp(file, true);
    isize n = qp.P.rows();
    isize n_eq_in = qp.A.rows();

    if (qp.skip) {
      std::cout << " path: " << qp.filename << " n: " << n
                << " n_eq+n_in: " << n_eq_in << " - skipping" << std::endl;
    } else {
      std::cout << " path: " << qp.filename << " n: " << n
                << " n_eq+n_in: " << n_eq_in << std::endl;
    }

    if (!qp.skip) {

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
      proxqp::dense::QP<T> qp{
        dim, n_eq, n_in, false, proxsuite::proxqp::DenseBackend::Automatic
      }; // creating QP object
      qp.init(H, g, A, b, C, l, u);

      qp.settings.eps_abs = 2e-8;
      qp.settings.eps_rel = 0;
      qp.settings.eps_primal_inf = 1e-12;
      qp.settings.eps_dual_inf = 1e-12;
      auto& eps = qp.settings.eps_abs;

      for (size_t it = 0; it < 2; ++it) {
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
        std::cout << "iter " << qp.results.info.iter << std::endl;
        CHECK(proxqp::dense::infty_norm(H * x + g + A.transpose() * y +
                                        C.transpose() * z) < 2 * eps);
        CHECK(proxqp::dense::infty_norm(A * x - b) > -eps);
        CHECK((C * x - l).minCoeff() > -eps);
        CHECK((C * x - u).maxCoeff() < eps);

        if (it > 0) {
          CHECK(qp.results.info.iter == 0);
        }
      }
      timer.stop();
      elapsed_time += timer.elapsed().user;
    }
  }
  std::cout << "timings total : \t" << elapsed_time * 1e-3 << "ms" << std::endl;
}
