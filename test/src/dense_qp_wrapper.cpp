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

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with inequality constraints"
  "and empty equality constraints")
{
  std::cout << "---testing sparse random strongly convex qp with inequality "
               "constraints "
               "and empty equality constraints---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(0);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;

  // Testing with empty but properly sized matrix A  of size (0, 10)
  std::cout << "Solving QP with" << std::endl;
  std::cout << "dim: " << dim << std::endl;
  std::cout << "n_eq: " << n_eq << std::endl;
  std::cout << "n_in: " << n_in << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "A.cols() :  " << qp_random.A.cols() << std::endl;
  std::cout << "A.rows() :  " << qp_random.A.rows() << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  qp.init(qp_random.H,
          qp_random.g,
          nullopt,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u);
  qp.solve();

  T pri_res =
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>();
  T dua_res = (qp_random.H * qp.results.x + qp_random.g +
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  // Testing with empty matrix A  of size (0, 0)
  qp_random.A = Eigen::MatrixXd();
  qp_random.b = Eigen::VectorXd();

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;

  std::cout << "Solving QP with" << std::endl;
  std::cout << "dim: " << dim << std::endl;
  std::cout << "n_eq: " << n_eq << std::endl;
  std::cout << "n_in: " << n_in << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "A.cols() :  " << qp_random.A.cols() << std::endl;
  std::cout << "A.rows() :  " << qp_random.A.rows() << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
             helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
              .lpNorm<Eigen::Infinity>();
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  // Testing with nullopt
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;

  qp3.init(qp_random.H,
           qp_random.g,
           nullopt,
           nullopt,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp3.solve();

  pri_res = (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
             helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
              .lpNorm<Eigen::Infinity>();
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}
DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update H")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update H---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating H" << std::endl;
  qp_random.H.setIdentity();
  qp.update(qp_random.H, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt);
  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update A")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update A---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;

  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating A" << std::endl;
  qp_random.A = utils::rand::sparse_matrix_rand_not_compressed<T>(
    n_eq, dim, sparsity_factor);
  qp.update(nullopt, nullopt, qp_random.A, nullopt, nullopt, nullopt, nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update C")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update C---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;

  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating C" << std::endl;
  qp_random.C = utils::rand::sparse_matrix_rand_not_compressed<T>(
    n_in, dim, sparsity_factor);
  qp.update(nullopt, nullopt, nullopt, nullopt, qp_random.C, nullopt, nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update b")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update b---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;

  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating b" << std::endl;
  auto x_sol = utils::rand::vector_rand<T>(dim);
  qp_random.b = qp_random.A * x_sol;
  qp.update(nullopt, nullopt, nullopt, qp_random.b, nullopt, nullopt, nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update u")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update u---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating b" << std::endl;
  auto x_sol = utils::rand::vector_rand<T>(dim);
  auto delta = utils::Vec<T>(n_in);
  for (isize i = 0; i < n_in; ++i) {
    delta(i) = utils::rand::uniform_rand();
  }

  qp_random.u = qp_random.C * x_sol + delta;
  qp.update(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, qp_random.u);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update g")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update g---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating g" << std::endl;
  auto g = utils::rand::vector_rand<T>(dim);

  qp_random.g = g;
  qp.update(nullopt, qp_random.g, nullopt, nullopt, nullopt, nullopt, nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and inequality "
  "constraints: test update H and A and b and u and l")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints: test update H and A and b and u and l---"
    << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp_random.H << std::endl;
  std::cout << "g :  " << qp_random.g << std::endl;
  std::cout << "A :  " << qp_random.A << std::endl;
  std::cout << "b :  " << qp_random.b << std::endl;
  std::cout << "C :  " << qp_random.C << std::endl;
  std::cout << "u :  " << qp_random.u << std::endl;
  std::cout << "l :  " << qp_random.l << std::endl;

  std::cout << "testing updating b" << std::endl;
  qp_random.H = utils::rand::sparse_positive_definite_rand_not_compressed<T>(
    dim, strong_convexity_factor, sparsity_factor);
  qp_random.A = utils::rand::sparse_matrix_rand_not_compressed<T>(
    n_eq, dim, sparsity_factor);
  auto x_sol = utils::rand::vector_rand<T>(dim);
  auto delta = utils::Vec<T>(n_in);
  for (proxqp::isize i = 0; i < n_in; ++i) {
    delta(i) = utils::rand::uniform_rand();
  }
  qp_random.b = qp_random.A * x_sol;
  qp_random.u = qp_random.C * x_sol + delta;
  qp_random.l = qp_random.C * x_sol - delta;
  qp.update(qp_random.H,
            nullopt,
            qp_random.A,
            qp_random.b,
            nullopt,
            qp_random.l,
            qp_random.u);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << qp.model.H << std::endl;
  std::cout << "g :  " << qp.model.g << std::endl;
  std::cout << "A :  " << qp.model.A << std::endl;
  std::cout << "b :  " << qp.model.b << std::endl;
  std::cout << "C :  " << qp.model.C << std::endl;
  std::cout << "u :  " << qp.model.u << std::endl;
  std::cout << "l :  " << qp.model.l << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update rho")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update rho---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "rho :  " << qp.results.info.rho << std::endl;

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            true,
            T(1.e-7),
            nullopt,
            nullopt); // restart the problem with default options
  std::cout << "after upating" << std::endl;
  std::cout << "rho :  " << qp.results.info.rho << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true,
           T(1.e-7),
           nullopt,
           nullopt);
  std::cout << "rho :  " << qp2.results.info.rho << std::endl;
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test update mu_eq and mu_in")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update mu_eq and mu_in---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "mu_in :  " << qp.results.info.mu_in << std::endl;
  std::cout << "mu_eq :  " << qp.results.info.mu_eq << std::endl;

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            true,
            nullopt,
            T(1.e-2),
            T(1.e-3));

  std::cout << "after upating" << std::endl;
  std::cout << "mu_in :  " << qp.results.info.mu_in << std::endl;
  std::cout << "mu_eq :  " << qp.results.info.mu_eq << std::endl;

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true,
           nullopt,
           T(1.e-2),
           T(1.e-3));
  qp2.solve();
  std::cout << "mu_in :  " << qp2.results.info.mu_in << std::endl;
  std::cout << "mu_eq :  " << qp2.results.info.mu_eq << std::endl;
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test warm starting")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test warm starting---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
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
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  auto x_wm = utils::rand::vector_rand<T>(dim);
  auto y_wm = utils::rand::vector_rand<T>(n_eq);
  auto z_wm = utils::rand::vector_rand<T>(n_in);
  std::cout << "proposed warm start" << std::endl;
  std::cout << "x_wm :  " << x_wm << std::endl;
  std::cout << "y_wm :  " << y_wm << std::endl;
  std::cout << "z_wm :  " << z_wm << std::endl;
  qp.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp.solve(x_wm, y_wm, z_wm);

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve(x_wm, y_wm, z_wm);

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test dense init")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test dense init---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      qp_random.H),
    qp_random.g,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      qp_random.A),
    qp_random.b,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      qp_random.C),
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
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test with no initial guess")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test with no initial guess---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
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
  std::cout << "------using API solving qp with dim with qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with qp2: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test with equality constrained initial guess")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints: test with equality constrained initial guess---"
    << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
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
  std::cout << "------using API solving qp with dim with qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with qp2: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test with warm start with previous result")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints: test with warm start with previous result---"
    << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
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
  std::cout << "------using API solving qp with dim with qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true);

  auto x = qp.results.x;
  auto y = qp.results.y;
  auto z = qp.results.z;
  // std::cout << "after scaling x " << x <<  " qp.results.x " << qp.results.x
  // << std::endl;
  qp2.ruiz.scale_primal_in_place({ from_eigen, x });
  qp2.ruiz.scale_dual_in_place_eq({ from_eigen, y });
  qp2.ruiz.scale_dual_in_place_in({ from_eigen, z });
  // std::cout << "after scaling x " << x <<  " qp.results.x " << qp.results.x
  // << std::endl;
  qp2.solve(x, y, z);

  qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp.update(
    nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, false);
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "------using API solving qp with dim with qp after warm start "
               "with previous result: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with qp2: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test with cold start option")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test with cold start option---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
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
  std::cout << "------using API solving qp with dim with qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true);

  auto x = qp.results.x;
  auto y = qp.results.y;
  auto z = qp.results.z;
  // std::cout << "after scaling x " << x <<  " qp.results.x " << qp.results.x
  // << std::endl;
  qp2.ruiz.scale_primal_in_place({ from_eigen, x });
  qp2.ruiz.scale_dual_in_place_eq({ from_eigen, y });
  qp2.ruiz.scale_dual_in_place_in({ from_eigen, z });
  // std::cout << "after scaling x " << x <<  " qp.results.x " << qp.results.x
  // << std::endl;
  qp2.solve(x, y, z);

  qp.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  qp.update(
    nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, true);
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "------using API solving qp with dim with qp after warm start "
               "with cold start option: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with cold start option: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test equilibration options at initialization")
{

  std::cout
    << "---testing sparse random strongly convex qp with equality and "
       "inequality constraints: test equilibration options at initialization---"
    << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          true);
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
  std::cout << "------using API solving qp with dim with qp with "
               "preconditioner derived: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "ruiz vector : " << qp.ruiz.delta << " ruiz scalar factor "
            << qp.ruiz.c << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           false);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp without preconditioner derivation: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "ruiz vector : " << qp2.ruiz.delta << " ruiz scalar factor "
            << qp2.ruiz.c << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test equilibration options at update")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test equilibration options at update---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          true);
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
  std::cout << "------using API solving qp with dim with qp with "
               "preconditioner derived: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            true); // rederive preconditioner with previous options, i.e., redo
                   // exact same derivations
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with preconditioner re derived "
               "after an update (should get exact same results): "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "------using API solving qp with preconditioner derivation and "
               "another object QP: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;

  qp2.update(
    nullopt,
    nullopt,
    nullopt,
    nullopt,
    nullopt,
    nullopt,
    nullopt,
    false); // use previous preconditioner: should get same result as well
  qp2.solve();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp without preconditioner derivation: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "ruiz vector : " << qp2.ruiz.delta << " ruiz scalar factor "
            << qp2.ruiz.c << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

/////  TESTS ALL INITIAL GUESS OPTIONS FOR MULTIPLE SOLVES AT ONCE
TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with no initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with equality "
          "constrained initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with equality constrained initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with equality "
          "constrained initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with cold start "
          "initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with cold start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with warm start")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start and first solve with no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess = InitialGuessStatus::WARM_START;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve(qp.results.x, qp.results.y, qp.results.z);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve(qp.results.x, qp.results.y, qp.results.z);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve(qp.results.x, qp.results.y, qp.results.z);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: warm start test from init")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start and first solve with no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2(dim, n_eq, n_in);
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  std::cout << "dirty workspace for qp2 : " << qp2.work.dirty << std::endl;
  qp2.solve(qp.results.x, qp.results.y, qp.results.z);
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve with new QP object" << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

/// TESTS WITH UPDATE + INITIAL GUESS OPTIONS

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test update and multiple solve at once with "
          "no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with no initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            update_preconditioner);
  std::cout << "dirty workspace after update : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "equality constrained initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with equality constrained initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            update_preconditioner);
  std::cout << "dirty workspace after update : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test update + multiple solve at once with equality "
  "constrained initial guess and then warm start with previous results")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            update_preconditioner);
  std::cout << "dirty workspace after update : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            update_preconditioner);
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "cold start initial guess and then cold start option")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with cold start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  qp.update(qp_random.H,
            qp_random.g,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            update_preconditioner);
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "warm start")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start and first solve with no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  qp.settings.initial_guess = InitialGuessStatus::WARM_START;
  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  auto x_wm = qp.results.x; // keep previous result
  auto y_wm = qp.results.y;
  auto z_wm = qp.results.z;
  bool update_preconditioner = true;
  // test with a false update (the warm start should give the exact solution)
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            update_preconditioner);
  std::cout << "dirty workspace after update: " << qp.work.dirty << std::endl;
  qp.solve(x_wm, y_wm, z_wm);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  x_wm = qp.results.x; // keep previous result
  y_wm = qp.results.y;
  z_wm = qp.results.z;
  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  // try now with a real update
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            update_preconditioner);
  std::cout << "dirty workspace after update: " << qp.work.dirty << std::endl;
  qp.solve(x_wm, y_wm, z_wm);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve(qp.results.x, qp.results.y, qp.results.z);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << qp.work.dirty << std::endl;
  qp.solve(qp.results.x, qp.results.y, qp.results.z);
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fifth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "ProxQP::dense: Test initializaton with rho for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test initializaton with rho for different initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          true,
          T(1.E-7));
  qp.solve();
  CHECK(qp.results.info.rho == T(1.E-7));
  T pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  T dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2(dim, n_eq, n_in);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true,
           T(1.E-7));
  qp2.solve();
  CHECK(qp2.results.info.rho == T(1.E-7));
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;

  dense::QP<T> qp3(dim, n_eq, n_in);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true,
           T(1.E-7));
  qp3.solve();
  CHECK(qp3.results.info.rho == T(1.E-7));
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp3.results.info.setup_time << " solve time "
            << qp3.results.info.solve_time << std::endl;

  dense::QP<T> qp4(dim, n_eq, n_in);
  qp4.settings.eps_abs = eps_abs;
  qp4.settings.eps_rel = 0;
  qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  qp4.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true,
           T(1.E-7));
  qp4.solve();
  CHECK(qp4.results.info.rho == T(1.E-7));
  pri_res = std::max(
    (qp_random.A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + qp_random.g +
             qp_random.A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp4.results.info.setup_time << " solve time "
            << qp4.results.info.solve_time << std::endl;

  dense::QP<T> qp5(dim, n_eq, n_in);
  qp5.settings.eps_abs = eps_abs;
  qp5.settings.eps_rel = 0;
  qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp5.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           true,
           T(1.E-7));
  qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z);
  CHECK(qp5.results.info.rho == T(1.E-7));
  pri_res = std::max(
    (qp_random.A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + qp_random.g +
             qp_random.A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp5.results.info.setup_time << " solve time "
            << qp5.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: Test g update for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test g update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  auto old_g = qp_random.g;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  qp.update(nullopt, qp_random.g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp.model.g - qp_random.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2(dim, n_eq, n_in);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2.init(qp_random.H,
           old_g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + old_g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp2.update(nullopt, qp_random.g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp2.model.g - qp_random.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;

  dense::QP<T> qp3(dim, n_eq, n_in);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp3.init(qp_random.H,
           old_g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + old_g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp3.update(nullopt, qp_random.g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp3.model.g - qp_random.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp3.results.info.setup_time << " solve time "
            << qp3.results.info.solve_time << std::endl;

  dense::QP<T> qp4(dim, n_eq, n_in);
  qp4.settings.eps_abs = eps_abs;
  qp4.settings.eps_rel = 0;
  qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  qp4.init(qp_random.H,
           old_g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp4.solve();
  pri_res = std::max(
    (qp_random.A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + old_g +
             qp_random.A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp4.update(nullopt, qp_random.g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp4.solve();
  pri_res = std::max(
    (qp_random.A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + qp_random.g +
             qp_random.A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp4.model.g - qp_random.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp4.results.info.setup_time << " solve time "
            << qp4.results.info.solve_time << std::endl;

  dense::QP<T> qp5(dim, n_eq, n_in);
  qp5.settings.eps_abs = eps_abs;
  qp5.settings.eps_rel = 0;
  qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp5.init(qp_random.H,
           old_g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z);
  pri_res = std::max(
    (qp_random.A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + old_g +
             qp_random.A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp5.update(nullopt, qp_random.g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp5.solve();
  pri_res = std::max(
    (qp_random.A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + qp_random.g +
             qp_random.A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp5.model.g - qp_random.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp5.results.info.setup_time << " solve time "
            << qp5.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: Test A update for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test A update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  auto new_A = utils::rand::sparse_matrix_rand_not_compressed<T>(
    n_eq, dim, sparsity_factor);
  qp.update(nullopt, nullopt, new_A, nullopt, nullopt, nullopt, nullopt);
  qp.solve();
  pri_res =
    std::max((new_A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
             (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
              helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
               .lpNorm<Eigen::Infinity>());
  dua_res =
    (qp_random.H * qp.results.x + qp_random.g +
     new_A.transpose() * qp.results.y + qp_random.C.transpose() * qp.results.z)
      .lpNorm<Eigen::Infinity>();
  CHECK((qp.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2(dim, n_eq, n_in);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp2.update(nullopt, nullopt, new_A, nullopt, nullopt, nullopt, nullopt);
  qp2.solve();
  pri_res = std::max(
    (new_A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             new_A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp2.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;

  dense::QP<T> qp3(dim, n_eq, n_in);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp3.update(nullopt, nullopt, new_A, nullopt, nullopt, nullopt, nullopt);
  qp3.solve();
  pri_res = std::max(
    (new_A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             new_A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp3.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp3.results.info.setup_time << " solve time "
            << qp3.results.info.solve_time << std::endl;

  dense::QP<T> qp4(dim, n_eq, n_in);
  qp4.settings.eps_abs = eps_abs;
  qp4.settings.eps_rel = 0;
  qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  qp4.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp4.solve();
  pri_res = std::max(
    (qp_random.A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + qp_random.g +
             qp_random.A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp4.update(nullopt, nullopt, new_A, nullopt, nullopt, nullopt, nullopt);
  qp4.solve();
  pri_res = std::max(
    (new_A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + qp_random.g +
             new_A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp4.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp4.results.info.setup_time << " solve time "
            << qp4.results.info.solve_time << std::endl;

  dense::QP<T> qp5(dim, n_eq, n_in);
  qp5.settings.eps_abs = eps_abs;
  qp5.settings.eps_rel = 0;
  qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp5.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z);
  pri_res = std::max(
    (qp_random.A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + qp_random.g +
             qp_random.A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp5.update(nullopt, nullopt, new_A, nullopt, nullopt, nullopt, nullopt);
  qp5.solve();
  pri_res = std::max(
    (new_A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + qp_random.g +
             new_A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((qp5.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp5.results.info.setup_time << " solve time "
            << qp5.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: Test rho update for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test rho update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            true,
            T(1.E-7));
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(qp.results.info.rho == T(1.E-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2(dim, n_eq, n_in);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp2.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             true,
             T(1.E-7));
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(qp2.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;

  dense::QP<T> qp3(dim, n_eq, n_in);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             true,
             T(1.E-7));
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(qp3.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp3.results.info.setup_time << " solve time "
            << qp3.results.info.solve_time << std::endl;

  dense::QP<T> qp4(dim, n_eq, n_in);
  qp4.settings.eps_abs = eps_abs;
  qp4.settings.eps_rel = 0;
  qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  qp4.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp4.solve();
  pri_res = std::max(
    (qp_random.A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + qp_random.g +
             qp_random.A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp4.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             true,
             T(1.E-7));
  qp4.solve();
  pri_res = std::max(
    (qp_random.A * qp4.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp4.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp4.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp4.results.x + qp_random.g +
             qp_random.A.transpose() * qp4.results.y +
             qp_random.C.transpose() * qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(qp4.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp4.results.info.setup_time << " solve time "
            << qp4.results.info.solve_time << std::endl;

  dense::QP<T> qp5(dim, n_eq, n_in);
  qp5.settings.eps_abs = eps_abs;
  qp5.settings.eps_rel = 0;
  qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  qp5.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp5.solve(qp3.results.x, qp3.results.y, qp3.results.z);
  pri_res = std::max(
    (qp_random.A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + qp_random.g +
             qp_random.A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  qp5.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             true,
             T(1.E-7));
  qp5.solve();
  pri_res = std::max(
    (qp_random.A * qp5.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp5.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp5.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp5.results.x + qp_random.g +
             qp_random.A.transpose() * qp5.results.y +
             qp_random.C.transpose() * qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(qp5.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp5.results.info.setup_time << " solve time "
            << qp5.results.info.solve_time << std::endl;
}

TEST_CASE("ProxQP::dense: Test g update for different warm start with previous "
          "result option")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;

  std::cout << "Test rho update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << qp.work.dirty
            << std::endl;

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  // a new linear cost slightly modified
  auto g = qp_random.g * 0.95;

  qp.update(nullopt, g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp.solve();
  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res =
    (qp_random.H * qp.results.x + g + qp_random.A.transpose() * qp.results.y +
     qp_random.C.transpose() * qp.results.z)
      .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;

  dense::QP<T> qp2(dim, n_eq, n_in);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2.init(qp_random.H,
           g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u);
  qp2.solve();
  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res =
    (qp_random.H * qp2.results.x + g + qp_random.A.transpose() * qp2.results.y +
     qp_random.C.transpose() * qp2.results.z)
      .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp2.results.info.setup_time << " solve time "
            << qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after updates using warm start with previous results")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "updates using warm start with previous results---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  qp.settings.initial_guess =
    proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;

  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  qp3.settings.initial_guess =
    proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after updates using cold start with previous results")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "updates using cold start with previous results---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess =
    proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.initial_guess =
    proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.initial_guess =
    proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after updates using equality constrained initial guess")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "updates using equality constrained initial guess---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess =
    proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.initial_guess =
    proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.initial_guess =
    proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after updates using no initial guess")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "updates using no initial guess---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::NO_INITIAL_GUESS);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.initial_guess = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::NO_INITIAL_GUESS);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp2.results.x + qp_random.g +
             qp_random.A.transpose() * qp2.results.y +
             qp_random.C.transpose() * qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.initial_guess = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::NO_INITIAL_GUESS);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);

  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
  qp3.solve();
  pri_res = std::max(
    (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp3.results.x + qp_random.g +
             qp_random.A.transpose() * qp3.results.y +
             qp_random.C.transpose() * qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after several solves using warm start with previous results")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "several solves using warm start with previous results---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  qp.settings.initial_guess =
    proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  qp2.settings.initial_guess =
    proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  for (isize iter = 0; iter < 10; ++iter) {
    // warm start with previous result used, hence if the qp is small and
    // simple, the parameters should not changed during first solve, and also
    // after as we start at the solution
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    qp2.solve();
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp2.results.x + qp_random.g +
               qp_random.A.transpose() * qp2.results.y +
               qp_random.C.transpose() * qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.settings.verbose = true;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);

  for (isize iter = 0; iter < 10; ++iter) {
    // warm start with previous result used, hence if the qp is small and
    // simple, the parameters should not changed during first solve, and also
    // after as we start at the solution
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  for (isize iter = 0; iter < 10; ++iter) {
    // warm start with previous result used, hence if the qp is small and
    // simple, the parameters should not changed during first solve, and also
    // after as we start at the solution
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after several solves using cold start with previous results")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "several solves using cold start with previous results---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess =
    proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.initial_guess =
    proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    qp2.solve();
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp2.results.x + qp_random.g +
               qp_random.A.transpose() * qp2.results.y +
               qp_random.C.transpose() * qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.initial_guess =
    proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);

  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }
}

DOCTEST_TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings after several solves "
  "using equality constrained initial guess")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "several solves using equality constrained initial guess---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess =
    proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.initial_guess =
    proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    qp2.solve();
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp2.results.x + qp_random.g +
               qp_random.A.transpose() * qp2.results.y +
               qp_random.C.transpose() * qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.initial_guess =
    proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);

  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }
}

DOCTEST_TEST_CASE(
  "ProxQP::dense: sparse random strongly convex qp with equality and "
  "inequality constraints: test changing default settings "
  "after several solves using no initial guess")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test changing default settings after "
               "several solves using no initial guess---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T rho(1.e-7);
  T mu_eq(1.e-4);
  bool compute_preconditioner = true;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.initial_guess = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  DOCTEST_CHECK(qp.settings.initial_guess ==
                proxqp::InitialGuessStatus::NO_INITIAL_GUESS);
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          rho);
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);
  qp.solve();
  DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) <= 1.E-9);
  DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) <= 1.E-9);

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

  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(rho - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(rho - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            compute_preconditioner,
            1.e-6);
  for (isize iter = 0; iter < 10; ++iter) {
    qp.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp.settings.default_rho) < 1.e-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp.results.info.rho) < 1.e-9);
    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp2{ dim, n_eq, n_in }; // creating QP object
  qp2.settings.initial_guess = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  DOCTEST_CHECK(qp2.settings.initial_guess ==
                proxqp::InitialGuessStatus::NO_INITIAL_GUESS);
  qp2.settings.eps_abs = eps_abs;
  qp2.settings.eps_rel = 0;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           nullopt,
           mu_eq);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
  qp2.solve();
  DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
  DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);

  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    qp2.solve();
    DOCTEST_CHECK(std::abs(mu_eq - qp2.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp2.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp2.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp2.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp2.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp2.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp2.results.x + qp_random.g +
               qp_random.A.transpose() * qp2.results.y +
               qp_random.C.transpose() * qp2.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> qp3{ dim, n_eq, n_in }; // creating QP object
  qp3.settings.initial_guess = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  DOCTEST_CHECK(qp3.settings.initial_guess ==
                proxqp::InitialGuessStatus::NO_INITIAL_GUESS);
  qp3.settings.eps_abs = eps_abs;
  qp3.settings.eps_rel = 0;
  qp3.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           compute_preconditioner,
           rho,
           mu_eq);

  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(rho - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(rho - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(mu_eq - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(T(1) / mu_eq - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }

  qp3.update(nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             nullopt,
             compute_preconditioner,
             1.e-6,
             1.e-3);
  for (isize iter = 0; iter < 10; ++iter) {
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    qp3.solve();
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.settings.default_rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-6 - qp3.results.info.rho) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.settings.default_mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e-3 - qp3.results.info.mu_eq) <= 1.E-9);
    DOCTEST_CHECK(std::abs(1.e3 - qp3.results.info.mu_eq_inv) <= 1.E-9);
    pri_res = std::max(
      (qp_random.A * qp3.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp3.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp3.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp3.results.x + qp_random.g +
               qp_random.A.transpose() * qp3.results.y +
               qp_random.C.transpose() * qp3.results.z)
                .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= eps_abs);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }
}

TEST_CASE("ProxQP::dense: init must be called before update")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  // call update without init, update calls init internally
  qp.update(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true);

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
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);

  qp_random.H *= 2.;
  qp_random.g = utils::rand::vector_rand<T>(dim);
  qp.update(qp_random.H,
            qp_random.g,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            true);

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  dua_res = (qp_random.H * qp.results.x + qp_random.g +
             qp_random.A.transpose() * qp.results.y +
             qp_random.C.transpose() * qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
}
// test of the box constraints interface
TEST_CASE("ProxQP::dense: check ordering of z when there are box constraints")
{
  dense::isize n_test(1000);
  double sparsity_factor = 1.;
  T eps_abs = T(1e-9);
  dense::isize dim = 15;

  // mixing ineq and box constraints
  for (isize i = 0; i < n_test; i++) {
    utils::rand::set_seed(i);
    dense::isize n_eq(dim / 4);
    dense::isize n_in(dim / 4);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    // ineq and boxes
    Eigen::Matrix<T, Eigen::Dynamic, 1> x_sol =
      utils::rand::vector_rand<T>(dim);
    Eigen::Matrix<T, Eigen::Dynamic, 1> delta(n_in);
    for (proxqp::isize i = 0; i < n_in; ++i) {
      delta(i) = utils::rand::uniform_rand();
    }
    qp_random.u = qp_random.C * x_sol + delta;
    qp_random.b = qp_random.A * x_sol;
    Eigen::Matrix<T, Eigen::Dynamic, 1> u_box(dim);
    u_box.setZero();
    Eigen::Matrix<T, Eigen::Dynamic, 1> l_box(dim);
    l_box.setZero();
    for (proxqp::isize i = 0; i < dim; ++i) {
      T shift = utils::rand::uniform_rand();
      u_box(i) = x_sol(i) + shift;
      l_box(i) = x_sol(i) - shift;
    }
    /////////////////// for debuging
    // using Mat =
    //   Eigen::Matrix<T,
    //                 Eigen::Dynamic,
    //                 Eigen::Dynamic, Eigen::ColMajor>;
    // Mat C_enlarged(dim+n_in,dim);
    // C_enlarged.setZero();
    // C_enlarged.topLeftCorner(n_in,dim) = qp_random.C;
    // C_enlarged.bottomLeftCorner(dim,dim).diagonal().array() += 1.;
    // Eigen::Matrix<T, Eigen::Dynamic, 1> u_enlarged(n_in+dim);
    // Eigen::Matrix<T, Eigen::Dynamic, 1> l_enlarged(n_in+dim);
    // u_enlarged.head(n_in) = qp_random.u;
    // u_enlarged.tail(dim) = u_box;
    // l_enlarged.head(n_in) = qp_random.l;
    // l_enlarged.tail(dim) = l_box;
    // std::cout << "n " << dim << " n_eq " << n_eq << " n_in "<< n_in <<
    // std::endl; std::cout << "=================qp compare" << std::endl;
    // proxqp::dense::QP<T> qp_compare{ dim, n_eq, dim + n_in, false};
    // qp_compare.settings.eps_abs = eps_abs;
    // qp_compare.settings.eps_rel = 0;
    // qp_compare.settings.max_iter = 10;
    // qp_compare.settings.max_iter_in = 10;
    // qp_compare.settings.verbose = true;
    // qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    // qp_compare.init(qp_random.H,
    //                 qp_random.g,
    //                 qp_random.A,
    //                 qp_random.b,
    //                 C_enlarged,
    //                 l_enlarged,
    //                 u_enlarged,
    //                 true);
    // qp_compare.solve();
    // std::cout << "=================qp compare end" << std::endl;
    ////////////////

    dense::QP<T> qp(dim, n_eq, n_in, true);
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            l_box,
            u_box,
            true);
    qp.solve();

    T pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    pri_res = std::max(pri_res,
                       (helpers::positive_part(qp.results.x - u_box) +
                        helpers::negative_part(qp.results.x - l_box))
                         .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp.results.x + qp_random.g +
                 qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z.head(n_in) +
                 qp.results.z.tail(dim))
                  .lpNorm<Eigen::Infinity>();
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
  }
  // idem but without ineq constraints
  for (isize i = 0; i < n_test; i++) {
    utils::rand::set_seed(i);
    dense::isize n_eq(dim / 4);
    dense::isize n_in(0);
    T strong_convexity_factor(1.e-2);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    // ineq and boxes
    Eigen::Matrix<T, Eigen::Dynamic, 1> x_sol =
      utils::rand::vector_rand<T>(dim);
    Eigen::Matrix<T, Eigen::Dynamic, 1> delta(n_in);
    for (proxqp::isize i = 0; i < n_in; ++i) {
      delta(i) = utils::rand::uniform_rand();
    }
    qp_random.u = qp_random.C * x_sol + delta;
    qp_random.b = qp_random.A * x_sol;
    Eigen::Matrix<T, Eigen::Dynamic, 1> u_box(dim);
    u_box.setZero();
    Eigen::Matrix<T, Eigen::Dynamic, 1> l_box(dim);
    l_box.setZero();
    for (proxqp::isize i = 0; i < dim; ++i) {
      T shift = utils::rand::uniform_rand();
      u_box(i) = x_sol(i) + shift;
      l_box(i) = x_sol(i) - shift;
    }

    dense::QP<T> qp(dim, n_eq, n_in, true);
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            l_box,
            u_box);

    qp.solve();

    T pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    pri_res = std::max(pri_res,
                       (helpers::positive_part(qp.results.x - u_box) +
                        helpers::negative_part(qp.results.x - l_box))
                         .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp.results.x + qp_random.g +
                 qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z.head(n_in) +
                 qp.results.z.tail(dim))
                  .lpNorm<Eigen::Infinity>();
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
  }
  // idem but without ineq and without eq constraints
  for (isize i = 0; i < n_test; i++) {
    dense::isize n_eq(0);
    dense::isize n_in(0);
    T strong_convexity_factor(1.e-2);
    using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Mat eye(dim, dim);
    eye.setZero();
    eye.diagonal().array() += 1.;

    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    // ineq and boxes
    Eigen::Matrix<T, Eigen::Dynamic, 1> x_sol =
      utils::rand::vector_rand<T>(dim);
    Eigen::Matrix<T, Eigen::Dynamic, 1> delta(n_in);
    for (proxqp::isize i = 0; i < n_in; ++i) {
      delta(i) = utils::rand::uniform_rand();
    }
    qp_random.u = qp_random.C * x_sol + delta;
    qp_random.b = qp_random.A * x_sol;
    Eigen::Matrix<T, Eigen::Dynamic, 1> u_box(dim);
    u_box.setZero();
    Eigen::Matrix<T, Eigen::Dynamic, 1> l_box(dim);
    l_box.setZero();
    for (proxqp::isize i = 0; i < dim; ++i) {
      T shift = utils::rand::uniform_rand();
      u_box(i) = x_sol(i) + shift;
      l_box(i) = x_sol(i) - shift;
    }
    // make a qp to compare
    dense::QP<T> qp_compare(dim, n_eq, dim, false);
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp_compare.settings.compute_preconditioner = true;
    qp_compare.init(qp_random.H,
                    qp_random.g,
                    qp_random.A,
                    qp_random.b,
                    eye,
                    l_box,
                    u_box,
                    true);

    qp_compare.solve();

    T pri_res = std::max(
      (qp_random.A * qp_compare.results.x - qp_random.b)
        .lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp_compare.results.x -
                              qp_random.u) +
       helpers::negative_part(qp_random.C * qp_compare.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    pri_res = std::max(pri_res,
                       (helpers::positive_part(qp_compare.results.x - u_box) +
                        helpers::negative_part(qp_compare.results.x - l_box))
                         .lpNorm<Eigen::Infinity>());
    T dua_res = (qp_random.H * qp_compare.results.x + qp_random.g +
                 qp_random.C.transpose() * qp_compare.results.z.head(n_in) +
                 qp_random.A.transpose() * qp_compare.results.y +
                 qp_compare.results.z.tail(dim))
                  .lpNorm<Eigen::Infinity>();
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
    // ineq and boxes
    dense::QP<T> qp(dim, n_eq, n_in, true);
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.settings.compute_preconditioner = true;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            l_box,
            u_box,
            true);

    qp.solve();

    pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    pri_res = std::max(pri_res,
                       (helpers::positive_part(qp.results.x - u_box) +
                        helpers::negative_part(qp.results.x - l_box))
                         .lpNorm<Eigen::Infinity>());
    dua_res = (qp_random.H * qp.results.x + qp_random.g +
               qp_random.A.transpose() * qp.results.y +
               qp_random.C.transpose() * qp.results.z.head(n_in) +
               qp.results.z.tail(dim))
                .lpNorm<Eigen::Infinity>();
    CHECK(dua_res <= eps_abs);
    CHECK(pri_res <= eps_abs);
  }
}
TEST_CASE("ProxQP::dense: check updates work when there are box constraints")
{

  double sparsity_factor = 1.;
  T eps_abs = T(1e-9);
  dense::isize dim = 50;
  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  // ineq and boxes
  dense::QP<T> qp(dim, n_eq, n_in, true);
  Eigen::Matrix<T, Eigen::Dynamic, 1> u_box(dim);
  u_box.setZero();
  u_box.array() += 1.E2;
  Eigen::Matrix<T, Eigen::Dynamic, 1> l_box(dim);
  l_box.setZero();
  l_box.array() -= 1.E2;
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          l_box,
          u_box);

  qp.solve();

  T pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  pri_res = std::max(pri_res,
                     (helpers::positive_part(qp.results.x - u_box) +
                      helpers::negative_part(qp.results.x - l_box))
                       .lpNorm<Eigen::Infinity>());
  T dua_res =
    (qp_random.H * qp.results.x + qp_random.g +
     qp_random.A.transpose() * qp.results.y +
     qp_random.C.transpose() * qp.results.z.head(n_in) + qp.results.z.tail(dim))
      .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);

  u_box.array() += 1.E1;
  l_box.array() -= 1.E1;

  qp.update(nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            qp_random.l,
            qp_random.u,
            l_box,
            u_box);

  qp.solve();

  pri_res = std::max(
    (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
    (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
     helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
      .lpNorm<Eigen::Infinity>());
  pri_res = std::max(pri_res,
                     (helpers::positive_part(qp.results.x - u_box) +
                      helpers::negative_part(qp.results.x - l_box))
                       .lpNorm<Eigen::Infinity>());
  dua_res =
    (qp_random.H * qp.results.x + qp_random.g +
     qp_random.A.transpose() * qp.results.y +
     qp_random.C.transpose() * qp.results.z.head(n_in) + qp.results.z.tail(dim))
      .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
}
TEST_CASE("ProxQP::dense: test primal infeasibility solving")
{
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-5);
  utils::rand::set_seed(1);
  dense::isize dim = 20;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    // create infeasible problem
    qp_random.b.array() += T(10.);
    qp_random.u.array() -= T(100.);
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.settings.primal_infeasibility_solving = true;
    qp.settings.eps_primal_inf = T(1.E-4);
    qp.settings.eps_dual_inf = T(1.E-4);
    qp.settings.verbose = true;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qp.solve();

    proxsuite::proxqp::utils::Vec<T> rhs_dim(dim);
    proxsuite::proxqp::utils::Vec<T> rhs_n_eq(n_eq);
    rhs_n_eq.setOnes();
    proxsuite::proxqp::utils::Vec<T> rhs_n_in(n_in);
    rhs_n_in.setOnes();
    rhs_dim.noalias() =
      qp_random.A.transpose() * rhs_n_eq + qp_random.C.transpose() * rhs_n_in;
    T scaled_eps = (rhs_dim).lpNorm<Eigen::Infinity>() * eps_abs;

    T pri_res =
      (qp_random.A.transpose() * (qp_random.A * qp.results.x - qp_random.b) +
       qp_random.C.transpose() *
         (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
          helpers::negative_part(qp_random.C * qp.results.x - qp_random.l)))
        .lpNorm<Eigen::Infinity>();
    T dua_res = (qp_random.H.selfadjointView<Eigen::Upper>() * qp.results.x +
                 qp_random.g + qp_random.A.transpose() * qp.results.y +
                 qp_random.C.transpose() * qp.results.z)
                  .lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK(pri_res <= scaled_eps);
    DOCTEST_CHECK(dua_res <= eps_abs);
  }
}

TEST_CASE("ProxQP::dense: estimate of minimal eigenvalues using Eigen")
{
  double sparsity_factor = 1.;
  T tol = T(1e-6);
  utils::rand::set_seed(1);
  dense::isize dim = 2;
  dense::isize n_eq(dim);
  dense::isize n_in(dim);
  T strong_convexity_factor(1.e-2);
  for (isize i = 0; i < 1; ++i) {
    // trivial test
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qp_random.H.setZero();
    qp_random.H.diagonal().setOnes();
    qp_random.H.diagonal().tail(1).setConstant(-1.);

    T estimate_minimal_eigen_value =
      dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        qp_random.H, EigenValueEstimateMethodOption::ExactMethod, 1.E-6, 10000);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            estimate_minimal_eigen_value);

    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate + 1) <=
                  tol);
  }
  dim = 50;
  n_eq = dim;
  n_in = dim;
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qp_random.H.setZero();
    dense::Vec<T> random_diag = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.H.diagonal().array() += random_diag.array();
    T minimal_eigenvalue = qp_random.H.diagonal().minCoeff();

    T estimate_minimal_eigen_value =
      dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        qp_random.H, EigenValueEstimateMethodOption::ExactMethod, 1.E-6, 10000);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            estimate_minimal_eigen_value);
    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate -
                           minimal_eigenvalue) <= tol);
  }
  dim = 50;
  n_eq = dim;
  n_in = dim;
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    dense::Vec<T> random_diag = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.H.diagonal().array() += 100 * random_diag.array();
    Eigen::SelfAdjointEigenSolver<dense::Mat<T>> es(qp_random.H,
                                                    Eigen::EigenvaluesOnly);
    T minimal_eigenvalue = T(es.eigenvalues().minCoeff());

    T estimate_minimal_eigen_value =
      dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        qp_random.H, EigenValueEstimateMethodOption::ExactMethod, 1.E-6, 10000);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            estimate_minimal_eigen_value);

    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate -
                           minimal_eigenvalue) <= tol);
  }
}

TEST_CASE(
  "ProxQP::dense: test estimate of minimal eigenvalue using manual choice")
{
  double sparsity_factor = 1.;
  T tol = T(1e-6);
  utils::rand::set_seed(1);
  dense::isize dim = 2;
  dense::isize n_eq(dim);
  dense::isize n_in(dim);
  T strong_convexity_factor(1.e-2);
  for (isize i = 0; i < 1; ++i) {
    // trivial test
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qp_random.H.setZero();
    qp_random.H.diagonal().setOnes();
    qp_random.H.diagonal().tail(1).setConstant(-1.);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            -1);

    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate + 1) <=
                  tol);
  }
  dim = 50;
  n_eq = dim;
  n_in = dim;
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qp_random.H.setZero();
    dense::Vec<T> random_diag = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.H.diagonal().array() += random_diag.array();
    T minimal_eigenvalue = qp_random.H.diagonal().minCoeff();

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            minimal_eigenvalue);
    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate -
                           minimal_eigenvalue) <= tol);
  }
  dim = 50;
  n_eq = dim;
  n_in = dim;
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    dense::Vec<T> random_diag = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.H.diagonal().array() += 100 * random_diag.array();
    Eigen::SelfAdjointEigenSolver<dense::Mat<T>> es(qp_random.H,
                                                    Eigen::EigenvaluesOnly);
    T minimal_eigenvalue = T(es.eigenvalues().minCoeff());

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            minimal_eigenvalue);

    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate -
                           minimal_eigenvalue) <= tol);
  }
}

TEST_CASE(
  "ProxQP::dense: test estimate of minimal eigenvalue using power iteration")
{
  double sparsity_factor = 1.;
  T tol = T(1e-3);
  utils::rand::set_seed(1);
  dense::isize dim = 2;
  dense::isize n_eq(dim);
  dense::isize n_in(dim);
  T strong_convexity_factor(1.e-2);
  for (isize i = 0; i < 1; ++i) {
    // trivial test
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qp_random.H.setZero();
    qp_random.H.diagonal().setOnes();
    qp_random.H.diagonal().tail(1).setConstant(-0.5);

    T estimate_minimal_eigen_value =
      dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        qp_random.H,
        EigenValueEstimateMethodOption::PowerIteration,
        1.E-6,
        10000);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            estimate_minimal_eigen_value);

    DOCTEST_CHECK(
      std::abs(qp.results.info.minimal_H_eigenvalue_estimate + 0.5) <= tol);
  }
  dim = 50;
  n_eq = dim;
  n_in = dim;
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qp_random.H.setZero();
    dense::Vec<T> random_diag = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.H.diagonal().array() += random_diag.array();
    T minimal_eigenvalue = qp_random.H.diagonal().minCoeff();

    T estimate_minimal_eigen_value =
      dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        qp_random.H,
        EigenValueEstimateMethodOption::PowerIteration,
        1.E-6,
        10000);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            estimate_minimal_eigen_value);
    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate -
                           minimal_eigenvalue) <= tol);
  }
  dim = 50;
  n_eq = dim;
  n_in = dim;
  for (isize i = 0; i < 20; ++i) {
    ::proxsuite::proxqp::utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    dense::Vec<T> random_diag = proxqp::utils::rand::vector_rand<T>(dim);
    qp_random.H.diagonal().array() +=
      100 * random_diag.array(); // add some random values to dense matrix
    Eigen::SelfAdjointEigenSolver<dense::Mat<T>> es(qp_random.H,
                                                    Eigen::EigenvaluesOnly);
    T minimal_eigenvalue = T(es.eigenvalues().minCoeff());

    T estimate_minimal_eigen_value =
      dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        qp_random.H,
        EigenValueEstimateMethodOption::PowerIteration,
        1.E-6,
        10000);

    proxqp::dense::QP<T> qp(dim, n_eq, n_in);
    qp.settings.max_iter = 1;
    qp.settings.max_iter_in = 1;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u,
            true,
            nullopt,
            nullopt,
            nullopt,
            estimate_minimal_eigen_value);

    DOCTEST_CHECK(std::abs(qp.results.info.minimal_H_eigenvalue_estimate -
                           minimal_eigenvalue) <= tol);
  }
}

DOCTEST_TEST_CASE("check that model.is_valid function for symmetric matrices "
                  "works for epsilon precision")
{
  Eigen::Matrix<T, 3, 3> matrix = Eigen::Matrix<T, 3, 3>::Random();
  Eigen::Matrix<T, 3, 3> symmetric_mat = matrix + matrix.transpose();

  symmetric_mat(0, 1) =
    symmetric_mat(1, 0) + std::numeric_limits<double>::epsilon();

  // compare the two checks for symmetry with and without tolerance
  bool is_symmetric_without_tolerance =
    symmetric_mat.isApprox(symmetric_mat.transpose(), 0.0);
  bool is_symmetric_with_tolerance = symmetric_mat.isApprox(
    symmetric_mat.transpose(),
    std::numeric_limits<typename decltype(symmetric_mat)::Scalar>::epsilon());
  DOCTEST_CHECK(is_symmetric_without_tolerance == false);
  DOCTEST_CHECK(is_symmetric_with_tolerance == true);

  // initialize a model with a symmetric matrix as Hessian, this runs
  // model.is_valid() that performs the check above
  proxqp::dense::QP<T> qp(3, 0, 0);
  qp.init(symmetric_mat, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt);
}

TEST_CASE("ProxQP::dense: test memory allocation when estimating biggest "
          "eigenvalue with power iteration")
{
  double sparsity_factor = 1.;
  utils::rand::set_seed(1);
  dense::isize dim = 2;
  dense::isize n_eq(dim);
  dense::isize n_in(dim);
  T strong_convexity_factor(1.e-2);
  Eigen::Matrix<double, 2, 2, Eigen::ColMajor> H;
  Eigen::VectorXd dw(2), rhs(2), err_v(2);
  // trivial test
  ::proxsuite::proxqp::utils::rand::set_seed(1234);
  proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  qp_random.H.setZero();
  qp_random.H.diagonal().setOnes();
  qp_random.H.diagonal().tail(1).setConstant(-0.5);
  H = qp_random.H;
  PROXSUITE_EIGEN_MALLOC_NOT_ALLOWED();
  dense::power_iteration(H, dw, rhs, err_v, 1.E-6, 10000);
  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}
