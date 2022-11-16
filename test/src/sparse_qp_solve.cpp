//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

using namespace proxsuite;
using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::utils;
using T = double;
using I = c_int;
using namespace proxsuite::linalg::sparse::tags;

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test solve function")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test solve function---"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    T eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);

    /*
    auto H = ::proxsuite::proxqp::utils::rand::sparse_positive_definite_rand(
      n, T(10.0), sparsity_factor);
    auto g = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    auto A = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_eq, n,
    sparsity_factor); auto x_sol =
    ::proxsuite::proxqp::utils::rand::vector_rand<T>(n); auto b = A * x_sol;
    auto C = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_in, n,
    sparsity_factor); auto l = C * x_sol; auto u = (l.array() +
    10).matrix().eval();

    utils::SparseRandomQP<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
                                  n,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);
    */
    proxqp::dense::Model<T> qp_dense = utils::dense_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxqp::sparse::SparseModel<T> qp = qp_dense.to_sparse();
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(qp.H,
                                             qp.g,
                                             qp.A,
                                             qp.b,
                                             qp.C,
                                             qp.l,
                                             qp.u,
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             eps_abs);

    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << n
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
                  "inequality constraints: test solve with different rho value")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test solve with different rho value---"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(qp.H,
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
                                             nullopt,
                                             T(1.E-7));
    DOCTEST_CHECK(results.info.rho == T(1.E-7));
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << results.info.iter
              << std::endl;
    std::cout << "setup timing " << results.info.setup_time << " solve time "
              << results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test solve with different mu_eq and mu_in values")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test solve with different mu_eq and "
               "mu_in values---"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(qp.H,
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
                                             nullopt,
                                             nullopt,
                                             T(1.E-2),
                                             T(1.E-2));
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << results.info.iter
              << std::endl;
    std::cout << "setup timing " << results.info.setup_time << " solve time "
              << results.info.solve_time << std::endl;
  }
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test setting specific sparse backend")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints: test setting specific sparse backend ---"
    << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxsuite::proxqp::InitialGuessStatus initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    proxsuite::proxqp::SparseBackend sparse_backend =
      proxsuite::proxqp::SparseBackend::MatrixFree;
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(qp.H,
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
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             true,
                                             true,
                                             nullopt,
                                             initial_guess,
                                             sparse_backend);
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
    DOCTEST_CHECK(results.info.sparse_backend == SparseBackend::MatrixFree);

    std::cout << "------using API solving qp with dim: " << n
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
                  "inequality constraints: test warm starting")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test warm starting---"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    auto x_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
    auto y_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n_eq);
    auto z_wm = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n_in);
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(
        qp.H, qp.g, qp.A, qp.b, qp.C, qp.l, qp.u, x_wm, y_wm, z_wm, eps_abs);
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << n
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
                  "inequality constraints: test verbose = true")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test verbose = true ---"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    bool verbose = true;
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(qp.H,
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
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             verbose);
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << n
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
                  "inequality constraints: test no initial guess")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test no initial guess ---"
            << std::endl;
  for (auto const& dims : { // proxsuite::linalg::veg::tuplify(50, 0, 0),
                            // proxsuite::linalg::veg::tuplify(50, 25, 0),
                            // proxsuite::linalg::veg::tuplify(10, 0, 10),
                            // proxsuite::linalg::veg::tuplify(50, 0, 25),
                            // proxsuite::linalg::veg::tuplify(50, 10, 25),
                            proxsuite::linalg::veg::tuplify(10, 3, 2) }) {
    VEG_BIND(auto const&, (n, n_eq, n_in), dims);

    double eps_abs = 1.e-9;
    T sparsity_factor = 0.15;
    T strong_convexity_factor = 0.01;
    ::proxsuite::proxqp::utils::rand::set_seed(1);
    proxqp::sparse::SparseModel<T> qp = utils::sparse_strongly_convex_qp(
      n, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    proxsuite::proxqp::InitialGuessStatus initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    proxsuite::proxqp::Results<T> results =
      proxsuite::proxqp::sparse::solve<T, I>(qp.H,
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
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             nullopt,
                                             true,
                                             true,
                                             nullopt,
                                             initial_guess);
    T dua_res = proxqp::dense::infty_norm(
      qp.H.selfadjointView<Eigen::Upper>() * results.x + qp.g +
      qp.A.transpose() * results.y + qp.C.transpose() * results.z);
    T pri_res = std::max(proxqp::dense::infty_norm(qp.A * results.x - qp.b),
                         proxqp::dense::infty_norm(
                           helpers::positive_part(qp.C * results.x - qp.u) +
                           helpers::negative_part(qp.C * results.x - qp.l)));
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------using API solving qp with dim: " << n
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << results.info.iter
              << std::endl;
    std::cout << "setup timing " << results.info.setup_time << " solve time "
              << results.info.solve_time << std::endl;
  }
}
