//
// Copyright (c) 2022-2023 INRIA
//
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using namespace proxsuite::proxqp;
using Scalar = double;

DOCTEST_TEST_CASE("ruiz preconditioner")
{
  int dim = 5;
  int n_eq = 6;
  int n_in = 0;
  auto sym = Symmetry::general; // 0 : upper triangular (by default),
  // 1:
  // auto sym = proxqp::Symmetry::lower; // 0 : upper triangular (by default),
  // 1: lower triangular ; else full matrix

  Scalar sparsity_factor(0.75);
  Scalar strong_convexity_factor(0.01);
  dense::Model<Scalar> qp_random = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  switch (sym) {
    case Symmetry::upper: {
      qp_random.H = qp_random.H.triangularView<Eigen::Upper>();
      break;
    }
    case Symmetry::lower: {
      qp_random.H = qp_random.H.triangularView<Eigen::Lower>();
      break;
    }
    default: {
    }
  }
  proxsuite::osqp::dense::QP<Scalar> qp{ dim,
                                         n_eq,
                                         n_in }; // creating QP object

  // Specific values for OSQP
  qp.settings.default_mu_eq = 1.e-2;
  qp.settings.default_mu_in = 1.e1;
  qp.settings.eps_abs = 1.e-4;
  qp.settings.eps_rel = 1.e-4;
  qp.settings.check_duality_gap = false;
  qp.settings.eps_duality_gap_abs = 1.e-3;
  qp.settings.eps_duality_gap_rel = 1.e-3;

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

// Settings test:
// RAS

// Note test:
// Passes