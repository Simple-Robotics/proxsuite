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

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and inequality constraints "
  "and increasing dimension using wrapper API")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints and increasing dimension using wrapper API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 4);
    proxqp::isize n_in(dim / 4);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

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

    std::cout << "------using API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with box inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---testing sparse random strongly convex qp with box inequality "
       "constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(0);
    proxqp::isize n_in(dim);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_box_constrained_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
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

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
              << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random not strongly convex qp with inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---testing sparse random not strongly convex qp with inequality "
       "constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {
    proxqp::isize n_in(dim / 2);
    proxqp::isize n_eq(0);
    proxqp::dense::Model<T> qp_random =
      proxqp::utils::dense_not_strongly_convex_qp(
        dim, n_eq, n_in, sparsity_factor);

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

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
              << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with degenerate inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---testing sparse random strongly convex qp with degenerate "
       "inequality constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.45;
  T eps_abs = T(1e-9);
  T strong_convexity_factor(1e-2);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {
    proxqp::isize m(dim / 4);
    proxqp::isize n_in(2 * m);
    proxqp::isize n_eq(0);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_degenerate_qp(
      dim,
      n_eq,
      m, // it n_in = 2 * m, it doubles the inequality constraints
      sparsity_factor,
      strong_convexity_factor);
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
    DOCTEST_CHECK(qp.results.info.status ==
                  proxqp::QPSolverOutput::PROXQP_SOLVED);
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

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
              << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}

DOCTEST_TEST_CASE("linear problem with equality inequality constraints and "
                  "increasing dimension using the API")
{
  srand(1);
  std::cout << "---testing linear problem with inequality constraints and "
               "increasing dimension using the API---"
            << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {
    proxqp::isize n_in(dim / 2);
    proxqp::isize n_eq(0);
    proxqp::dense::Model<T> qp_random =
      proxqp::utils::dense_not_strongly_convex_qp(
        dim, n_eq, n_in, sparsity_factor);
    qp_random.H.setZero();
    auto z_sol = proxqp::utils::rand::vector_rand<T>(n_in);
    qp_random.g = -qp_random.C.transpose() *
                  z_sol; // make sure the LP is bounded within the feasible set
    // std::cout << "g : " << qp.g << " C " << qp.C  << " u " << qp.u << " l "
    // << qp.l << std::endl;
    proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.settings.verbose = false;
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

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
              << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
}
