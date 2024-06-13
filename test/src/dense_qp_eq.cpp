//
// Copyright (c) 2022 - 2024 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using T = double;
using namespace proxsuite;
DOCTEST_TEST_CASE("qp: start from solution using the wrapper framework")
{
  proxqp::isize dim = 30;
  proxqp::isize n_eq = 6;
  proxqp::isize n_in = 0;
  T sparsity_factor = 0.15;
  T strong_convexity_factor(1.e-2);
  std::cout << "---testing sparse random strongly convex qp with equality "
               "constraints and starting at the solution using the wrapper "
               "framework---"
            << std::endl;
  proxqp::utils::rand::set_seed(1);
  auto H = ::proxsuite::proxqp::utils::rand::
    sparse_positive_definite_rand_not_compressed(
      dim, strong_convexity_factor, sparsity_factor);
  auto A =
    ::proxsuite::proxqp::utils::rand::sparse_matrix_rand_not_compressed<T>(
      n_eq, dim, sparsity_factor);
  auto solution = ::proxsuite::proxqp::utils::rand::vector_rand<T>(dim + n_eq);
  auto primal_solution = solution.topRows(dim);
  auto dual_solution = solution.bottomRows(n_eq);
  auto b = A * primal_solution;
  auto g = -H * primal_solution - A.transpose() * dual_solution;
  auto C =
    ::proxsuite::proxqp::utils::rand::sparse_matrix_rand_not_compressed<T>(
      0, dim, sparsity_factor);
  Eigen::Matrix<T, Eigen::Dynamic, 1> dual_init_in(n_in);
  Eigen::Matrix<T, Eigen::Dynamic, 1> u(0);
  Eigen::Matrix<T, Eigen::Dynamic, 1> l(0);
  dual_init_in.setZero();
  T eps_abs = T(1e-9);

  proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START;
  qp.init(H, g, A, b, C, l, u);
  qp.solve(primal_solution, dual_solution, dual_init_in);

  DOCTEST_CHECK((A * qp.results.x - b).lpNorm<Eigen::Infinity>() <= eps_abs);
  DOCTEST_CHECK((H * qp.results.x + g + A.transpose() * qp.results.y)
                  .lpNorm<Eigen::Infinity>() <= eps_abs);
}
DOCTEST_TEST_CASE("sparse random strongly convex qp with equality constraints "
                  "and increasing dimension with the wrapper API")
{

  std::cout << "---testing sparse random strongly convex qp with equality "
               "constraints and increasing dimension with the wrapper API---"
            << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 2);
    proxqp::isize n_in(0);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.u,
            qp_random.l);
    qp.solve();
    T pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp.results.x + qp_random.g +
                 qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using wrapper API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}
DOCTEST_TEST_CASE("linear problem with equality  with equality constraints and "
                  "linar cost and increasing dimension using wrapper API")
{

  std::cout << "---testing linear problem with equality constraints and "
               "increasing dimension using wrapper API---"
            << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 2);
    proxqp::isize n_in(0);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp_random.H.setZero();
    auto y_sol = proxqp::utils::rand::vector_rand<T>(
      n_eq); // make sure the LP is bounded within the feasible set
    qp_random.g = -qp_random.A.transpose() * y_sol;

    proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qp.solve();

    T pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp.results.x + qp_random.g +
                 qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using wrapper API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}

DOCTEST_TEST_CASE("linear problem with equality with equality constraints and "
                  "linear cost and increasing dimension using wrapper API and  "
                  "the dedicated LP interface")
{

  std::cout
    << "---testing LP interface for solving linear problem with "
       "equality constraints and increasing dimension using wrapper API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 2);
    proxqp::isize n_in(0);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp_random.H.setZero();
    auto y_sol = proxqp::utils::rand::vector_rand<T>(
      n_eq); // make sure the LP is bounded within the feasible set
    qp_random.g = -qp_random.A.transpose() * y_sol;

    proxqp::dense::QP<T> qp{
      dim, n_eq, n_in, proxqp::HessianType::Zero
    }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qp.solve();

    T pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp.results.x + qp_random.g +
                 qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using wrapper API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}

DOCTEST_TEST_CASE("infeasible qp")
{
  // (x1- 9)^2 + (x2-6)^2
  // s.t.
  // x1 <= 10
  // x2 <= 10
  // x1 >= 20
  Eigen::Matrix<T, 2, 2> H;
  H << 1.0, 0.0, 0.0, 1.0;
  H = 2 * H;

  Eigen::Matrix<T, 2, 1> g;
  g << -18.0, -12.0;

  Eigen::Matrix<T, 3, 2> C;
  C << 1, 0, // x1 <= 10
    0, 1,    // x2 <= 10
    -1, 0;   // x1 >= 20

  Eigen::Matrix<T, 3, 1> u;
  u << 10, 10, -20;

  int n = H.rows();
  int n_in = C.rows();
  int n_eq = 0;

  Eigen::Matrix<T, Eigen::Dynamic, 1> l =
    Eigen::Matrix<T, Eigen::Dynamic, 1>::Constant(
      n_in, -std::numeric_limits<double>::infinity());

  proxsuite::proxqp::dense::QP<T> qp(n, n_eq, n_in);
  qp.init(H, g, nullopt, nullopt, C, l, u);
  qp.settings.eps_rel = 0.;
  qp.settings.eps_abs = 1e-9;

  qp.solve();

  DOCTEST_CHECK(qp.results.info.status ==
                proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE);
}