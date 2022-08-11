//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <iostream>

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

  proxqp::dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START;
  Qp.init(H, g, A, b, C, u, l);
  Qp.solve(primal_solution, dual_solution, dual_init_in);

  DOCTEST_CHECK((A * Qp.results.x - b).lpNorm<Eigen::Infinity>() <= eps_abs);
  DOCTEST_CHECK((H * Qp.results.x + g + A.transpose() * Qp.results.y)
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
    proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxqp::dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();
    T pri_res =
      std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
               (proxqp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
                proxqp::dense::negative_part(qp.C * Qp.results.x - qp.l))
                 .lpNorm<Eigen::Infinity>());
    T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
                 qp.C.transpose() * Qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using wrapper API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
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
    proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp.H.setZero();
    auto y_sol = proxqp::utils::rand::vector_rand<T>(
      n_eq); // make sure the LP is bounded within the feasible set
    qp.g = -qp.A.transpose() * y_sol;

    proxqp::dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.settings.eps_rel = 0;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();

    T pri_res =
      std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
               (proxqp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
                proxqp::dense::negative_part(qp.C * Qp.results.x - qp.l))
                 .lpNorm<Eigen::Infinity>());
    T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
                 qp.C.transpose() * Qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using wrapper API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter
              << std::endl;
  }
}
