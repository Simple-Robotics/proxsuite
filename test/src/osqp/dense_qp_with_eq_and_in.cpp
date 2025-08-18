//
// Copyright (c) 2025 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/common/utils/random_qp_problems.hpp>

using T = double;
using namespace proxsuite;
using namespace proxsuite::common;

DOCTEST_TEST_CASE("OSQP:  sparse random strongly convex qp with equality and "
                  "inequality constraints "
                  "and increasing dimension using wrapper API")
{

  std::cout
    << "---OSQP:  testing sparse random strongly convex qp with equality and "
       "inequality constraints and increasing dimension using wrapper API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  common::utils::rand::set_seed(1);
  for (isize dim = 10; dim < 1000; dim += 100) {

    isize n_eq(dim / 4);
    isize n_in(dim / 4);
    T strong_convexity_factor(1.e-2);
    common::dense::Model<T> qp_random = common::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    osqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
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
    std::cout << "total number of iteration: " << qp.results.info.iter_ext
              << std::endl;
  }
}

DOCTEST_TEST_CASE("OSQP:  sparse random strongly convex qp with box inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---OSQP:  testing sparse random strongly convex qp with box inequality "
       "constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  common::utils::rand::set_seed(1);
  for (isize dim = 10; dim < 1000; dim += 100) {

    isize n_eq(0);
    isize n_in(dim);
    T strong_convexity_factor(1.e-2);
    common::dense::Model<T> qp_random = common::utils::dense_box_constrained_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    osqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
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
    std::cout << "total number of iteration: " << qp.results.info.iter_ext
              << std::endl;
  }
}

DOCTEST_TEST_CASE("OSQP:  sparse random not strongly convex qp with inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---OSQP:  testing sparse random not strongly convex qp with inequality "
       "constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  common::utils::rand::set_seed(1);
  for (isize dim = 10; dim < 1000; dim += 100) {
    isize n_in(dim / 2);
    isize n_eq(0);
    common::dense::Model<T> qp_random =
      common::utils::dense_not_strongly_convex_qp(
        dim, n_eq, n_in, sparsity_factor);

    osqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
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
    std::cout << "total number of iteration: " << qp.results.info.iter_ext
              << std::endl;
  }
}

DOCTEST_TEST_CASE(
  "OSQP:  sparse random strongly convex qp with degenerate inequality "
  "constraints and increasing dimension using the API")
{

  std::cout
    << "---OSQP:  testing sparse random strongly convex qp with degenerate "
       "inequality constraints and increasing dimension using the API---"
    << std::endl;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  T eps_primal_inf = T(1e-15);
  T eps_dual_inf = T(1e-15);
  T sparsity_factor = 0.45;
  T strong_convexity_factor(1e-2);
  common::utils::rand::set_seed(1);
  for (isize dim = 10; dim < 1000; dim += 100) {
    isize m(dim / 4);
    isize n_in(2 * m);
    isize n_eq(0);
    common::dense::Model<T> qp_random = common::utils::dense_degenerate_qp(
      dim,
      n_eq,
      m, // it n_in = 2 * m, it doubles the inequality constraints
      sparsity_factor,
      strong_convexity_factor);
    osqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
    qp.settings.eps_primal_inf = eps_primal_inf;
    qp.settings.eps_dual_inf = eps_dual_inf;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qp.solve();
    DOCTEST_CHECK(qp.results.info.status ==
                  common::QPSolverOutput::QPSOLVER_SOLVED);
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
    std::cout << "total number of iteration: " << qp.results.info.iter_ext
              << std::endl;
  }
  // Note:
  // eps_primal_inf and eps_dual_inf are set to 1e-15 to reproduce the
  // benchmark setting on OSQP in the benchmark repository:
  // https://github.com/Simple-Robotics/proxqp_benchmark
}

DOCTEST_TEST_CASE(
  "OSQP:  linear problem with equality inequality constraints and "
  "increasing dimension using the API")
{
  srand(1);
  std::cout
    << "---OSQP:  testing linear problem with inequality constraints and "
       "increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  common::utils::rand::set_seed(1);
  for (isize dim = 10; dim < 1000; dim += 100) {
    isize n_in(dim / 2);
    isize n_eq(0);
    common::dense::Model<T> qp_random =
      common::utils::dense_not_strongly_convex_qp(
        dim, n_eq, n_in, sparsity_factor);
    qp_random.H.setZero();
    auto z_sol = common::utils::rand::vector_rand<T>(n_in);
    qp_random.g = -qp_random.C.transpose() *
                  z_sol; // make sure the LP is bounded within the feasible set
    // std::cout << "g : " << qp.g << " C " << qp.C  << " u " << qp.u << " l "
    // << qp.l << std::endl;
    osqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
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
    std::cout << "total number of iteration: " << qp.results.info.iter_ext
              << std::endl;
  }
}

DOCTEST_TEST_CASE(
  "OSQP:  sparse random strongly convex qp with equality and inequality "
  "constraints "
  "and increasing dimension using wrapper API to test different settings "
  "on solution polishing.")
{

  std::cout
    << "---OSQP:  testing sparse random strongly convex qp with equality and "
       "inequality constraints and increasing dimension using wrapper API "
       "to test different settings on solution polishing---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  common::utils::rand::set_seed(1);
  for (isize dim = 10; dim < 1000; dim += 100) {

    isize n_eq(dim / 4);
    isize n_in(dim / 4);
    T strong_convexity_factor(1.e-2);
    common::dense::Model<T> qp_random = common::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    // Trivial test
    osqp::dense::QP<T> qp{ dim, n_eq, n_in };
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
    qp.settings.polish = true;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);

    DOCTEST_CHECK(qp.results.info.status_polish ==
                  common::PolishStatus::POLISH_NOT_RUN);

    qp.solve();

    DOCTEST_CHECK(qp.results.info.status_polish !=
                  common::PolishStatus::POLISH_NO_ACTIVE_SET_FOUND);

    // Polishing not run because problem is not solved as
    // algorithm is stopped early
    osqp::dense::QP<T> qp2{ dim, n_eq, n_in };
    qp2.settings.eps_abs = eps_abs;
    qp2.settings.eps_rel = eps_rel;
    qp2.settings.polish = true;
    qp2.settings.max_iter = 1;
    qp2.init(qp_random.H,
             qp_random.g,
             qp_random.A,
             qp_random.b,
             qp_random.C,
             qp_random.l,
             qp_random.u);

    DOCTEST_CHECK(qp2.results.info.status_polish ==
                  common::PolishStatus::POLISH_NOT_RUN);

    qp2.solve();

    DOCTEST_CHECK(qp2.results.info.status_polish ==
                  common::PolishStatus::POLISH_NOT_RUN);

    // Polish succeeds as the problem is not hard (compared
    // to some Maros Meszaros ones, see OSQP benchmarks)
    osqp::dense::QP<T> qp3{ dim, n_eq, n_in };
    qp3.settings.eps_abs = eps_abs;
    qp3.settings.eps_rel = eps_rel;
    qp3.settings.polish = true;
    qp3.init(qp_random.H,
             qp_random.g,
             qp_random.A,
             qp_random.b,
             qp_random.C,
             qp_random.l,
             qp_random.u);

    DOCTEST_CHECK(qp3.results.info.status_polish ==
                  common::PolishStatus::POLISH_NOT_RUN);

    qp3.solve();

    DOCTEST_CHECK(qp3.results.info.status_polish ==
                  common::PolishStatus::POLISH_SUCCEEDED);
  }
}
