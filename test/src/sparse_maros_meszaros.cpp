//
// Copyright (c) 2022-2023 INRIA
//

#include <iostream>
#include <doctest.hpp>
#include <maros_meszaros.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>

using namespace proxsuite;

#define MAROS_MESZAROS_DIR PROBLEM_PATH "/data/maros_meszaros_data/"

template<typename T>
void
compute_primal_dual_feasibility(const PreprocessedQpSparse& preprocessed,
                                const proxqp::Results<T>& results,
                                T& primal_feasibility,
                                T& dual_feasibility)
{
  dual_feasibility = proxsuite::proxqp::dense::infty_norm(
    preprocessed.H.selfadjointView<Eigen::Upper>() * results.x +
    preprocessed.g + preprocessed.AT * results.y + preprocessed.CT * results.z);

  T prim_eq = proxsuite::proxqp::dense::infty_norm(
    preprocessed.AT.transpose() * results.x - preprocessed.b);
  T prim_in =
    std::max(proxsuite::proxqp::dense::infty_norm(
               preprocessed.AT.transpose() * results.x - preprocessed.b),
             proxsuite::proxqp::dense::infty_norm(
               helpers::positive_part(preprocessed.CT.transpose() * results.x -
                                      preprocessed.u) +
               helpers::negative_part(preprocessed.CT.transpose() * results.x -
                                      preprocessed.l)));
  primal_feasibility = std::max(prim_eq, prim_in);
}

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

TEST_CASE("sparse maros meszaros using the API")
{
  using isize = proxsuite::proxqp::sparse::isize;
  using T = double;
  using I = mat_int32_t;

  const T eps_abs_no_duality_gap = 2e-8;
  const T eps_abs_with_duality_gap = 1e-5;

  for (auto const* file : files) {
    auto qp_raw = load_qp(file);
    isize n = qp_raw.P.rows();
    isize n_eq_in = qp_raw.A.rows();

    bool skip = (n > 1000 || n_eq_in > 1000);
    if (skip) {
      std::cout << " path: " << qp_raw.filename << " n: " << n
                << " n_eq+n_in: " << n_eq_in << " - SKIPPING" << std::endl;
    } else {
      std::cout << " path: " << qp_raw.filename << " n: " << n
                << " n_eq+n_in: " << n_eq_in << std::endl;
    }

    if (!skip) {

      auto preprocessed = preprocess_qp_sparse(VEG_FWD(qp_raw));
      auto& H = preprocessed.H;
      auto& AT = preprocessed.AT;
      auto& CT = preprocessed.CT;
      auto& g = preprocessed.g;
      auto& b = preprocessed.b;
      auto& u = preprocessed.u;
      auto& l = preprocessed.l;

      isize n_in = CT.cols();
      proxsuite::proxqp::sparse::QP<T, I> qp(H.cast<bool>(),
                                             AT.transpose().cast<bool>(),
                                             CT.transpose().cast<bool>());
      qp.settings.eps_abs = eps_abs_no_duality_gap;
      qp.settings.eps_rel = 0;
      qp.settings.eps_primal_inf = 1e-12;
      qp.settings.eps_dual_inf = 1e-12;
      qp.init(H, g, AT.transpose(), b, CT.transpose(), l, u);

      for (isize iter = 0; iter < 2; ++iter) {
        if (iter > 0)
          qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::
            WARM_START_WITH_PREVIOUS_RESULT;
        qp.solve();

        T primal_feasibility, dual_feasibility;
        compute_primal_dual_feasibility(
          preprocessed, qp.results, primal_feasibility, dual_feasibility);

        CHECK(primal_feasibility < qp.settings.eps_abs);
        CHECK(dual_feasibility < qp.settings.eps_abs);
        CHECK(qp.results.info.pri_res < eps_abs_with_duality_gap);
        CHECK(qp.results.info.dua_res < eps_abs_with_duality_gap);

        std::cout << "dual feasibility: " << dual_feasibility << std::endl;

        std::cout << "primal residual " << primal_feasibility << std::endl;
      }
      {
        qp.solve();

        CHECK(proxsuite::proxqp::dense::infty_norm(
                H.selfadjointView<Eigen::Upper>() * qp.results.x + g +
                AT * qp.results.y + CT * qp.results.z) <=
              eps_abs_no_duality_gap);
        CHECK(proxsuite::proxqp::dense::infty_norm(
                AT.transpose() * qp.results.x - b) <= eps_abs_no_duality_gap);
        if (n_in > 0) {
          CHECK((CT.transpose() * qp.results.x - l).minCoeff() >
                -eps_abs_no_duality_gap);
          CHECK((CT.transpose() * qp.results.x - u).maxCoeff() <
                eps_abs_no_duality_gap);
        }

        T primal_feasibility, dual_feasibility;
        compute_primal_dual_feasibility(
          preprocessed, qp.results, primal_feasibility, dual_feasibility);

        std::cout << "dual residual " << dual_feasibility << std::endl;

        std::cout << "primal residual " << primal_feasibility << std::endl;
      }

      {
        qp.settings.initial_guess =
          proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
        qp.settings.check_duality_gap = true;
        qp.settings.eps_abs = eps_abs_with_duality_gap;
        qp.settings.eps_duality_gap_abs = eps_abs_with_duality_gap;

        qp.solve();

        T primal_feasibility, dual_feasibility;
        compute_primal_dual_feasibility(
          preprocessed, qp.results, primal_feasibility, dual_feasibility);

        CHECK(primal_feasibility < eps_abs_with_duality_gap);
        CHECK(dual_feasibility < eps_abs_with_duality_gap);
        CHECK(qp.results.info.duality_gap < eps_abs_with_duality_gap);
      }
    }
  }
}
