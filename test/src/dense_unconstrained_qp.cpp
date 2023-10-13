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
using namespace proxsuite;

using T = double;

DOCTEST_TEST_CASE(
  "sparse random strongly convex unconstrained qp and increasing dimension")
{

  std::cout << "---testing sparse random strongly convex qp with increasing "
               "dimension---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  for (int dim = 10; dim < 1000; dim += 100) {

    int n_eq(0);
    int n_in(0);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_unconstrained_qp(
      dim, sparsity_factor, strong_convexity_factor);
    proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
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

DOCTEST_TEST_CASE("sparse random not strongly convex unconstrained qp and "
                  "increasing dimension")
{

  std::cout << "---testing sparse random not strongly convex unconstrained qp "
               "with increasing dimension---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  for (int dim = 10; dim < 1000; dim += 100) {

    int n_eq(0);
    int n_in(0);
    T strong_convexity_factor(0);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_unconstrained_qp(
      dim, sparsity_factor, strong_convexity_factor);
    auto x_sol = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.g =
      -qp_random.H *
      x_sol; // to be dually feasible g must be in the image space of H

    proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    qp.settings.eps_abs = eps_abs;
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

DOCTEST_TEST_CASE("unconstrained qp with H = Id and g random")
{

  std::cout << "---unconstrained qp with H = Id and g random---" << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);

  int dim(100);
  int n_eq(0);
  int n_in(0);
  T strong_convexity_factor(1.E-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_unconstrained_qp(
    dim, sparsity_factor, strong_convexity_factor);
  qp_random.H.setZero();
  qp_random.H.diagonal().array() += 1;

  proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
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

DOCTEST_TEST_CASE("unconstrained qp with H = Id and g = 0")
{

  std::cout << "---unconstrained qp with H = Id and g = 0---" << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);

  int dim(100);
  int n_eq(0);
  int n_in(0);
  T strong_convexity_factor(1.E-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_unconstrained_qp(
    dim, sparsity_factor, strong_convexity_factor);
  qp_random.H.setZero();
  qp_random.H.diagonal().array() += 1;
  qp_random.g.setZero();

  proxqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
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
