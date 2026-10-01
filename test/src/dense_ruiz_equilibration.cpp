//
// Copyright (c) 2022-2023 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using namespace proxsuite;
using Scalar = double;

DOCTEST_TEST_CASE("ruiz preconditioner")
{
  int dim = 5;
  int n_eq = 6;
  int n_in = 0;
  auto sym = proxqp::Symmetry::general; // 0 : upper triangular (by default),
  // 1:
  // auto sym = proxqp::Symmetry::lower; // 0 : upper triangular (by default),
  // 1: lower triangular ; else full matrix

  Scalar sparsity_factor(0.75);
  Scalar strong_convexity_factor(0.01);
  proxqp::dense::Model<Scalar> qp_random =
    proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  switch (sym) {
    case proxqp::Symmetry::upper: {
      qp_random.H = qp_random.H.triangularView<Eigen::Upper>();
      break;
    }
    case proxqp::Symmetry::lower: {
      qp_random.H = qp_random.H.triangularView<Eigen::Lower>();
      break;
    }
    default: {
    }
  }
  proxqp::dense::QP<Scalar> qp{ dim, n_eq, n_in }; // creating QP object
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u);

  auto head = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
    qp.ruiz.delta.head(dim).asDiagonal());
  auto tail = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
    qp.ruiz.delta.tail(n_eq).asDiagonal());
  auto c = qp.ruiz.c;

  auto const& H = qp_random.H;
  auto const& g = qp_random.g;
  auto const& A = qp_random.A;
  auto const& b = qp_random.b;

  auto H_new = (c * head * H * head).eval();
  auto g_new = (c * head * g).eval();
  auto A_new = (tail * A * head).eval();
  auto b_new = (tail * b).eval();

  DOCTEST_CHECK((H_new - qp.work.H_scaled).norm() <= Scalar(1e-10));
  DOCTEST_CHECK((g_new - qp.work.g_scaled).norm() <= Scalar(1e-10));
  DOCTEST_CHECK((A_new - qp.work.A_scaled).norm() <= Scalar(1e-10));
  DOCTEST_CHECK((b_new - qp.work.b_scaled).norm() <= Scalar(1e-10));
}

DOCTEST_TEST_CASE(
  "ruiz preconditioner keeps the scaled box identity consistent")
{
  // Regression test: when update_preconditioner is false, the scaled identity
  // of the box constraint block (work.i_scaled) used to be multiplied by delta
  // again at every update instead of being derived from the unscaled model.
  // The box multipliers were then off by the accumulated factor, and the error
  // grew at every update even though the model never changed.
  int dim = 20;
  int n_eq = 0;
  int n_in = 0;

  Scalar sparsity_factor(0.75);
  Scalar strong_convexity_factor(0.01);
  proxqp::dense::Model<Scalar> qp_random =
    proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  // box bounds tight enough to be active at the solution: the box always
  // contains the origin, so the problem stays feasible
  auto x_unconstrained = (-qp_random.H.ldlt().solve(qp_random.g)).eval();
  Scalar bound = Scalar(0.5) * x_unconstrained.cwiseAbs().maxCoeff();
  auto l_box =
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Constant(dim, -bound).eval();
  auto u_box =
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Constant(dim, bound).eval();

  proxqp::dense::QP<Scalar> qp{ dim, n_eq, n_in, true };
  qp.settings.eps_abs = Scalar(1e-9);
  qp.settings.eps_rel = Scalar(0);
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          l_box,
          u_box);
  qp.solve();

  DOCTEST_CHECK(qp.results.info.status ==
                proxqp::QPSolverOutput::PROXQP_SOLVED);
  // at least one box constraint has to be active for this test to be
  // meaningful
  DOCTEST_CHECK(qp.results.z.tail(dim).lpNorm<Eigen::Infinity>() > Scalar(0));

  auto i_scaled_expected = qp.work.i_scaled.eval();

  // the model is never modified, so every update below is a no-op and has to
  // return the very same primal-dual solution
  for (int iter = 0; iter < 5; ++iter) {
    qp.update(qp_random.H,
              qp_random.g,
              qp_random.A,
              qp_random.b,
              qp_random.C,
              qp_random.l,
              qp_random.u,
              l_box,
              u_box,
              false);
    qp.solve();

    DOCTEST_CHECK(
      (qp.work.i_scaled - i_scaled_expected).lpNorm<Eigen::Infinity>() <=
      Scalar(1e-14));

    // stationarity of the returned primal-dual solution, box multipliers
    // included
    auto dual_residual =
      (qp_random.H * qp.results.x + qp_random.g + qp.results.z.tail(dim))
        .eval();
    DOCTEST_CHECK(dual_residual.lpNorm<Eigen::Infinity>() <= Scalar(1e-8));
  }
}
