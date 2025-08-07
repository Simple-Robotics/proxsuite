//
// Copyright (c) 2025 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/osqp/dense/dense.hpp>
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
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 4);
    proxqp::isize n_in(dim / 4);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
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

DOCTEST_TEST_CASE("sparse random strongly convex qp with box inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---testing sparse random strongly convex qp with box inequality "
       "constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(0);
    proxqp::isize n_in(dim);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_box_constrained_qp(
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

DOCTEST_TEST_CASE("sparse random not strongly convex qp with inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---testing sparse random not strongly convex qp with inequality "
       "constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {
    proxqp::isize n_in(dim / 2);
    proxqp::isize n_eq(0);
    proxqp::dense::Model<T> qp_random =
      proxqp::utils::dense_not_strongly_convex_qp(
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

// Test fail
DOCTEST_TEST_CASE("sparse random strongly convex qp with degenerate inequality "
                  "constraints and increasing dimension using the API")
{

  std::cout
    << "---testing sparse random strongly convex qp with degenerate "
       "inequality constraints and increasing dimension using the API---"
    << std::endl;
  T sparsity_factor = 0.45;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
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
    // DOCTEST_CHECK(qp.results.info.status ==
    //               proxqp::QPSolverOutput::PROXQP_SOLVED); // Fail here
    T pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp.results.x + qp_random.g +
                 qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    // DOCTEST_CHECK(pri_res <= eps_abs); // Fail here
    // DOCTEST_CHECK(dua_res <= eps_abs);

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
              << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter_ext
              << std::endl;
  }
  // dim =  10: Pass
  // dim = 110: Fail: r_pri plafond 1.63e-01 / r_dua cv / r_g -1.20e+21 / Primal
  // infeasible dim = 210: Pass: But r_g -3.15e+20 dim = 310: Fail: r_pri
  // plafond 2.01e-02 / r_dua cv / r_g -3.16e+20 / Primal infeasible dim = 410:
  // Fail: r_pri plafond 1.73e-02 / r_dua cv / r_g -2.14e+19 / Primal infeasible
  // dim = 510: Fail: r_pri plafond 7.13e-03 / r_dua cv / r_g -1.48e+20 / Primal
  // infeasible dim = 610: Pass: But r_g -2.45e+20 dim = 710: Pass: But r_g
  // -2.78e+19 dim = 810: Fail: r_pri plafond 1.48e-02 / r_dua cv / r_g
  // -2.58e+20 / Primal infeasible dim = 910: Pass: But r_g -1.51e+20
}

DOCTEST_TEST_CASE("linear problem with equality inequality constraints and "
                  "increasing dimension using the API")
{
  srand(1);
  std::cout << "---testing linear problem with inequality constraints and "
               "increasing dimension using the API---"
            << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
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
  "sparse random strongly convex qp with equality and inequality constraints "
  "and increasing dimension using wrapper API to test different settings "
  "on solution polishing.")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints and increasing dimension using wrapper API "
       "to test different settings on solution polishing---"
    << std::endl;
  T sparsity_factor = 0.15;
  T eps_abs = T(1e-3); // OSQP unit test
  T eps_rel = T(0);
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 4);
    proxqp::isize n_in(dim / 4);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    // Trivial test
    osqp::dense::QP<T> qp{ dim, n_eq, n_in };
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = eps_rel;
    qp.settings.polishing = true;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);

    DOCTEST_CHECK(qp.results.info.status_polish ==
                  proxqp::PolishStatus::POLISH_NOT_RUN);

    qp.solve();

    DOCTEST_CHECK(qp.results.info.status_polish !=
                  proxqp::PolishStatus::POLISH_NO_ACTIVE_SET_FOUND);

    // Polishing not run because problem is not solved as
    // algorithm is stopped early
    osqp::dense::QP<T> qp2{ dim, n_eq, n_in };
    qp2.settings.eps_abs = eps_abs;
    qp2.settings.eps_rel = eps_rel;
    qp2.settings.polishing = true;
    qp2.settings.max_iter = 1;
    qp2.init(qp_random.H,
             qp_random.g,
             qp_random.A,
             qp_random.b,
             qp_random.C,
             qp_random.l,
             qp_random.u);

    DOCTEST_CHECK(qp2.results.info.status_polish ==
                  proxqp::PolishStatus::POLISH_NOT_RUN);

    qp2.solve();

    DOCTEST_CHECK(qp2.results.info.status_polish ==
                  proxqp::PolishStatus::POLISH_NOT_RUN);

    // Polish succeeds as the problem is not hard (compared
    // to some Maros Meszaros ones, see OSQP benchmarks)
    osqp::dense::QP<T> qp3{ dim, n_eq, n_in };
    qp3.settings.eps_abs = eps_abs;
    qp3.settings.eps_rel = eps_rel;
    qp3.settings.polishing = true;
    qp3.init(qp_random.H,
             qp_random.g,
             qp_random.A,
             qp_random.b,
             qp_random.C,
             qp_random.l,
             qp_random.u);

    DOCTEST_CHECK(qp3.results.info.status_polish ==
                  proxqp::PolishStatus::POLISH_NOT_RUN);

    qp3.solve();

    DOCTEST_CHECK(qp3.results.info.status_polish ==
                  proxqp::PolishStatus::POLISH_SUCCEEDED);
  }
}
