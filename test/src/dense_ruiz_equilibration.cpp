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
