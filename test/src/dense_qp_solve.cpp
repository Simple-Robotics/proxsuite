//
// Copyright (c) 2022 INRIA
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
using namespace proxsuite::proxqp;

DOCTEST_TEST_CASE("proxqp::dense: test init with fixed sizes matrices")
{
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(5), n_in(2);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  Eigen::Matrix<T, 10, 10> H = qp.H;
  Eigen::Matrix<T, 10, 1> g = qp.g;
  Eigen::Matrix<T, 5, 10> A = qp.A;
  Eigen::Matrix<T, 5, 1> b = qp.b;
  Eigen::Matrix<T, 2, 10> C = qp.C;
  Eigen::Matrix<T, 2, 1> l = qp.l;
  Eigen::Matrix<T, 2, 1> u = qp.u;

  {
    Results<T> results = dense::solve<T>(
      H, g, A, b, C, l, u, nullopt, nullopt, nullopt, eps_abs, 0);

    T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                         (helpers::positive_part(qp.C * results.x - qp.u) +
                          helpers::negative_part(qp.C * results.x - qp.l))
                           .lpNorm<Eigen::Infinity>());
    T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
                 qp.C.transpose() * results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << results.info.iter
              << std::endl;
    std::cout << "setup timing " << results.info.setup_time << " solve time "
              << results.info.solve_time << std::endl;
  }

  {
    dense::QP<T> qp_problem(dim, n_eq, 0);
    qp_problem.init(H, g, A, b, nullopt, nullopt, nullopt);
    qp_problem.settings.eps_abs = eps_abs;
    qp_problem.solve();

    const Results<T>& results = qp_problem.results;

    T pri_res = (qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>();
    T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << results.info.iter
              << std::endl;
    std::cout << "setup timing " << results.info.setup_time << " solve time "
              << results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test solve function")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test solve function---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  Results<T> results = dense::solve<T>(qp.H,
                                       qp.g,
                                       qp.A,
                                       qp.b,
                                       qp.C,
                                       qp.l,
                                       qp.u,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       eps_abs,
                                       0);

  T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (helpers::positive_part(qp.C * results.x - qp.u) +
                        helpers::negative_part(qp.C * results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
               qp.C.transpose() * results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test solve with different rho value")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test solve with different rho value---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  Results<T> results = dense::solve<T>(qp.H,
                                       qp.g,
                                       qp.A,
                                       qp.b,
                                       qp.C,
                                       qp.l,
                                       qp.u,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       eps_abs,
                                       0,
                                       T(1.E-7));
  DOCTEST_CHECK(results.info.rho == T(1.E-7));
  T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (helpers::positive_part(qp.C * results.x - qp.u) +
                        helpers::negative_part(qp.C * results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
               qp.C.transpose() * results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test solve with different mu_eq and mu_in values")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test solve with different mu_eq and "
               "mu_in values---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  Results<T> results = dense::solve<T>(qp.H,
                                       qp.g,
                                       qp.A,
                                       qp.b,
                                       qp.C,
                                       qp.l,
                                       qp.u,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       eps_abs,
                                       0,
                                       nullopt,
                                       T(1.E-2),
                                       T(1.E-2));
  T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (helpers::positive_part(qp.C * results.x - qp.u) +
                        helpers::negative_part(qp.C * results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
               qp.C.transpose() * results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test warm starting")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test warm starting---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  auto x_wm = utils::rand::vector_rand<T>(dim);
  auto y_wm = utils::rand::vector_rand<T>(n_eq);
  auto z_wm = utils::rand::vector_rand<T>(n_in);
  Results<T> results = dense::solve<T>(
    qp.H, qp.g, qp.A, qp.b, qp.C, qp.l, qp.u, x_wm, y_wm, z_wm, eps_abs, 0);
  T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (helpers::positive_part(qp.C * results.x - qp.u) +
                        helpers::negative_part(qp.C * results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
               qp.C.transpose() * results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test verbose = true")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test verbose = true ---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  bool verbose = true;
  Results<T> results = dense::solve<T>(qp.H,
                                       qp.g,
                                       qp.A,
                                       qp.b,
                                       qp.C,
                                       qp.l,
                                       qp.u,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       eps_abs,
                                       0,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       verbose);
  T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (helpers::positive_part(qp.C * results.x - qp.u) +
                        helpers::negative_part(qp.C * results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
               qp.C.transpose() * results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test no initial guess")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test no initial guess ---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  InitialGuessStatus initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  Results<T> results = dense::solve<T>(qp.H,
                                       qp.g,
                                       qp.A,
                                       qp.b,
                                       qp.C,
                                       qp.l,
                                       qp.u,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       eps_abs,
                                       0,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       nullopt,
                                       true,
                                       true,
                                       nullopt,
                                       initial_guess);
  T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (helpers::positive_part(qp.C * results.x - qp.u) +
                        helpers::negative_part(qp.C * results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
               qp.C.transpose() * results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}
