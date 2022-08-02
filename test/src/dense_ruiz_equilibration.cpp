#include <doctest.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using namespace proxsuite;
using Scalar = long double;

DOCTEST_TEST_CASE("ruiz preconditioner")
{
  int dim = 5;
  int n_eq = 6;
  int n_in = 0;
  auto sym = proxqp::Symmetry::upper; // 0 : upper triangular (by default), 1:
                                      // lower triangular ; else full matrix

  Scalar sparsity_factor(0.15);
  Scalar strong_convexity_factor(0.01);
  proxqp::dense::Model<Scalar> qp = proxqp::utils::dense_strongly_convex_qp(
                                  dim,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

  switch (sym) {
    case proxqp::Symmetry::upper: {
      qp.H = qp.H.triangularView<Eigen::Upper>();
      break;
    }
    case proxqp::Symmetry::lower: {
      qp.H = qp.H.triangularView<Eigen::Lower>();
      break;
    }
    default: {
    }
  }

  proxqp::dense::QP<Scalar> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);

  auto head = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
    Qp.ruiz.delta.head(dim).asDiagonal());
  auto tail = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
    Qp.ruiz.delta.tail(n_eq).asDiagonal());
  auto c = Qp.ruiz.c;

  auto const& H = qp.H;
  auto const& g = qp.g;
  auto const& A = qp.A;
  auto const& b = qp.b;

  auto H_new = (c * head * H * head).eval();
  auto g_new = (c * head * g).eval();
  auto A_new = (tail * A * head).eval();
  auto b_new = (tail * b).eval();

  DOCTEST_CHECK((H_new - Qp.work.H_scaled).norm() <= Scalar(1e-10));
  DOCTEST_CHECK((g_new - Qp.work.g_scaled).norm() <= Scalar(1e-10));
  DOCTEST_CHECK((A_new - Qp.work.A_scaled).norm() <= Scalar(1e-10));
  DOCTEST_CHECK((b_new - Qp.work.b_scaled).norm() <= Scalar(1e-10));
}
