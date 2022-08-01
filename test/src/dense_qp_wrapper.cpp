//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <util.hpp>

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update H")
{
  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update H---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating H" << std::endl;
  qp.H.setIdentity();
  Qp.update(qp.H,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);
  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update A")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update A---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;

  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating A" << std::endl;
  qp.A = test::rand::sparse_matrix_rand_not_compressed<T>(
    n_eq, dim, sparsity_factor);
  Qp.update(std::nullopt,
            std::nullopt,
            qp.A,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update C")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update C---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;

  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating C" << std::endl;
  qp.C = test::rand::sparse_matrix_rand_not_compressed<T>(
    n_in, dim, sparsity_factor);
  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            qp.C,
            std::nullopt,
            std::nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update b")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update b---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;

  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating b" << std::endl;
  auto x_sol = test::rand::vector_rand<T>(dim);
  qp.b = qp.A * x_sol;
  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            qp.b,
            std::nullopt,
            std::nullopt,
            std::nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update u")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update u---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating b" << std::endl;
  auto x_sol = test::rand::vector_rand<T>(dim);
  auto delta = test::Vec<T>(n_in);
  for (isize i = 0; i < n_in; ++i) {
    delta(i) = test::rand::uniform_rand();
  }

  qp.u = qp.C * x_sol + delta;
  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            qp.u,
            std::nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update g")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update g---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating g" << std::endl;
  auto g = test::rand::vector_rand<T>(dim);

  qp.g = g;
  Qp.update(std::nullopt,
            qp.g,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
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
  test::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "H :  " << qp.H << std::endl;
  std::cout << "g :  " << qp.g << std::endl;
  std::cout << "A :  " << qp.A << std::endl;
  std::cout << "b :  " << qp.b << std::endl;
  std::cout << "C :  " << qp.C << std::endl;
  std::cout << "u :  " << qp.u << std::endl;
  std::cout << "l :  " << qp.l << std::endl;

  std::cout << "testing updating b" << std::endl;
  qp.H = test::rand::sparse_positive_definite_rand_not_compressed<T>(
    dim, strong_convexity_factor, sparsity_factor);
  qp.A = test::rand::sparse_matrix_rand_not_compressed<T>(
    n_eq, dim, sparsity_factor);
  auto x_sol = test::rand::vector_rand<T>(dim);
  auto delta = test::Vec<T>(n_in);
  for (proxqp::isize i = 0; i < n_in; ++i) {
    delta(i) = test::rand::uniform_rand();
  }
  qp.b = qp.A * x_sol;
  qp.u = qp.C * x_sol + delta;
  qp.l = qp.C * x_sol - delta;
  Qp.update(qp.H, std::nullopt, qp.A, qp.b, std::nullopt, qp.u, qp.l);

  std::cout << "after upating" << std::endl;
  std::cout << "H :  " << Qp.model.H << std::endl;
  std::cout << "g :  " << Qp.model.g << std::endl;
  std::cout << "A :  " << Qp.model.A << std::endl;
  std::cout << "b :  " << Qp.model.b << std::endl;
  std::cout << "C :  " << Qp.model.C << std::endl;
  std::cout << "u :  " << Qp.model.u << std::endl;
  std::cout << "l :  " << Qp.model.l << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update rho")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update rho---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "rho :  " << Qp.results.info.rho << std::endl;

  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            true,
            T(1.e-7),
            std::nullopt,
            std::nullopt); // restart the problem with default options
  std::cout << "after upating" << std::endl;
  std::cout << "rho :  " << Qp.results.info.rho << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H,
           qp.g,
           qp.A,
           qp.b,
           qp.C,
           qp.u,
           qp.l,
           true,
           T(1.e-7),
           std::nullopt,
           std::nullopt);
  std::cout << "rho :  " << Qp2.results.info.rho << std::endl;
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update mu_eq and mu_in")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test update mu_eq and mu_in---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "before upating" << std::endl;
  std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;
  std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;

  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            true,
            std::nullopt,
            T(1.e-2),
            T(1.e-3));

  std::cout << "after upating" << std::endl;
  std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;
  std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;

  Qp.solve();

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.init(qp.H,
           qp.g,
           qp.A,
           qp.b,
           qp.C,
           qp.u,
           qp.l,
           true,
           std::nullopt,
           T(1.e-2),
           T(1.e-3));
  Qp2.solve();
  std::cout << "mu_in :  " << Qp2.results.info.mu_in << std::endl;
  std::cout << "mu_eq :  " << Qp2.results.info.mu_eq << std::endl;
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test warm starting")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test warm starting---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  auto x_wm = test::rand::vector_rand<T>(dim);
  auto y_wm = test::rand::vector_rand<T>(n_eq);
  auto z_wm = test::rand::vector_rand<T>(n_in);
  std::cout << "proposed warm start" << std::endl;
  std::cout << "x_wm :  " << x_wm << std::endl;
  std::cout << "y_wm :  " << y_wm << std::endl;
  std::cout << "z_wm :  " << z_wm << std::endl;
  Qp.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp.solve(x_wm, y_wm, z_wm);

  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------using API solving qp with dim after updating: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  // conter factual check with another QP object starting at the updated model
  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve(x_wm, y_wm, z_wm);

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "------ conter factual check with another QP object starting at "
               "the updated model : "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test dense init")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test dense init---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      qp.H),
    qp.g,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      qp.A),
    qp.b,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      qp.C),
    qp.u,
    qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with no initial guess")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test with no initial guess---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp2: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
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
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();

  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp2: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
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
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);

  auto x = Qp.results.x;
  auto y = Qp.results.y;
  auto z = Qp.results.z;
  // std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x
  // << std::endl;
  Qp2.ruiz.scale_primal_in_place({ from_eigen, x });
  Qp2.ruiz.scale_dual_in_place_eq({ from_eigen, y });
  Qp2.ruiz.scale_dual_in_place_in({ from_eigen, z });
  // std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x
  // << std::endl;
  Qp2.solve(x, y, z);

  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            false);
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "------using API solving qp with dim with Qp after warm start "
               "with previous result: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp2: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with cold start option")
{

  std::cout << "---testing sparse random strongly convex qp with equality and "
               "inequality constraints: test with cold start option---"
            << std::endl;
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp: " << dim
            << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);

  auto x = Qp.results.x;
  auto y = Qp.results.y;
  auto z = Qp.results.z;
  // std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x
  // << std::endl;
  Qp2.ruiz.scale_primal_in_place({ from_eigen, x });
  Qp2.ruiz.scale_dual_in_place_eq({ from_eigen, y });
  Qp2.ruiz.scale_dual_in_place_in({ from_eigen, z });
  // std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x
  // << std::endl;
  Qp2.solve(x, y, z);

  Qp.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            true);
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "------using API solving qp with dim with Qp after warm start "
               "with cold start option: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with cold start option: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
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
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp with "
               "preconditioner derived: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "ruiz vector : " << Qp.ruiz.delta << " ruiz scalar factor "
            << Qp.ruiz.c << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, false);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp without preconditioner derivation: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "ruiz vector : " << Qp2.ruiz.delta << " ruiz scalar factor "
            << Qp2.ruiz.c << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
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
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp{ dim, n_eq, n_in }; // creating QP object
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);
  Qp.solve();
  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with dim with Qp with "
               "preconditioner derived: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            true); // rederive preconditioner with previous options, i.e., redo
                   // exact same derivations
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp with preconditioner re derived "
               "after an update (should get exact same results): "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2{ dim, n_eq, n_in }; // creating QP object
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "------using API solving qp with preconditioner derivation and "
               "another object QP: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;

  Qp2.update(
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    false); // use previous preconditioner: should get same result as well
  Qp2.solve();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);
  std::cout << "------using API solving qp without preconditioner derivation: "
            << dim << " neq: " << n_eq << " nin: " << n_in << std::endl;
  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "ruiz vector : " << Qp2.ruiz.delta << " ruiz scalar factor "
            << Qp2.ruiz.c << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

/////  TESTS ALL INITIAL GUESS OPTIONS FOR MULTIPLE SOLVES AT ONCE
TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with no initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with equality "
          "constrained initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with equality constrained initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with equality "
          "constrained initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);
  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with cold start "
          "initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with cold start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test multiple solve at once with warm start")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start and first solve with no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess = InitialGuessStatus::WARM_START;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: warm start test from init")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start and first solve with no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2(dim, n_eq, n_in);
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp2.settings.initial_guess = InitialGuessStatus::WARM_START;
  std::cout << "dirty workspace for Qp2 : " << Qp2.work.dirty << std::endl;
  Qp2.solve(Qp.results.x, Qp.results.y, Qp.results.z);
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve with new QP object" << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}

/// TESTS WITH UPDATE + INITIAL GUESS OPTIONS

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update and multiple solve at once with "
          "no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with no initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  qp.H *= 2.;
  qp.g = test::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
  std::cout << "dirty workspace after update : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "equality constrained initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with equality constrained initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  qp.H *= 2.;
  qp.g = test::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
  std::cout << "dirty workspace after update : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test update + multiple solve at once with equality "
  "constrained initial guess and then warm start with previous results")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  qp.H *= 2.;
  qp.g = test::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
  std::cout << "dirty workspace after update : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE(
  "sparse random strongly convex qp with equality and "
  "inequality constraints: test multiple solve at once with no initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start with previous result and first solve with "
               "no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  qp.H *= 2.;
  qp.g = test::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "cold start initial guess and then cold start option")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;

  std::cout << "Test with cold start with previous result and first solve with "
               "equality constrained initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  qp.H *= 2.;
  qp.g = test::rand::vector_rand<T>(dim);
  bool update_preconditioner = true;
  Qp.update(qp.H,
            qp.g,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            update_preconditioner);
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("sparse random strongly convex qp with equality and "
          "inequality constraints: test update + multiple solve at once with "
          "warm start")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test with warm start and first solve with no initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  Qp.settings.initial_guess = InitialGuessStatus::WARM_START;
  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  auto x_wm = Qp.results.x; // keep previous result
  auto y_wm = Qp.results.y;
  auto z_wm = Qp.results.z;
  bool update_preconditioner = true;
  // test with a false update (the warm start should give the exact solution)
  Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
  std::cout << "dirty workspace after update: " << Qp.work.dirty << std::endl;
  Qp.solve(x_wm, y_wm, z_wm);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  std::cout << "Second solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  x_wm = Qp.results.x; // keep previous result
  y_wm = Qp.results.y;
  z_wm = Qp.results.z;
  qp.H *= 2.;
  qp.g = test::rand::vector_rand<T>(dim);
  // try now with a real update
  Qp.update(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, update_preconditioner);
  std::cout << "dirty workspace after update: " << Qp.work.dirty << std::endl;
  Qp.solve(x_wm, y_wm, z_wm);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Third solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fourth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  std::cout << "dirty workspace : " << Qp.work.dirty << std::endl;
  Qp.solve(Qp.results.x, Qp.results.y, Qp.results.z);
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "Fifth solve " << std::endl;
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;
}

TEST_CASE("Test initializaton with rho for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test initializaton with rho for different initial guess"
            << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
  Qp.solve();
  CHECK(Qp.results.info.rho == T(1.E-7));
  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2(dim, n_eq, n_in);
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
  Qp2.solve();
  CHECK(Qp2.results.info.rho == T(1.E-7));
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;

  dense::QP<T> Qp3(dim, n_eq, n_in);
  Qp3.settings.eps_abs = eps_abs;
  Qp3.settings.eps_rel = 0;
  Qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
  Qp3.solve();
  CHECK(Qp3.results.info.rho == T(1.E-7));
  pri_res = std::max((qp.A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + qp.g + qp.A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp3.results.info.setup_time << " solve time "
            << Qp3.results.info.solve_time << std::endl;

  dense::QP<T> Qp4(dim, n_eq, n_in);
  Qp4.settings.eps_abs = eps_abs;
  Qp4.settings.eps_rel = 0;
  Qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
  Qp4.solve();
  CHECK(Qp4.results.info.rho == T(1.E-7));
  pri_res = std::max((qp.A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + qp.g + qp.A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp4.results.info.setup_time << " solve time "
            << Qp4.results.info.solve_time << std::endl;

  dense::QP<T> Qp5(dim, n_eq, n_in);
  Qp5.settings.eps_abs = eps_abs;
  Qp5.settings.eps_rel = 0;
  Qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, T(1.E-7));
  Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
  CHECK(Qp5.results.info.rho == T(1.E-7));
  pri_res = std::max((qp.A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + qp.g + qp.A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp5.results.info.setup_time << " solve time "
            << Qp5.results.info.solve_time << std::endl;
}

TEST_CASE("Test g update for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test g update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  auto old_g = qp.g;
  qp.g = test::rand::vector_rand<T>(dim);
  Qp.update(std::nullopt,
            qp.g,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp.model.g - qp.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2(dim, n_eq, n_in);
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Qp2.init(qp.H, old_g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + old_g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp2.update(std::nullopt,
             qp.g,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp2.model.g - qp.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;

  dense::QP<T> Qp3(dim, n_eq, n_in);
  Qp3.settings.eps_abs = eps_abs;
  Qp3.settings.eps_rel = 0;
  Qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp3.init(qp.H, old_g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp3.solve();
  pri_res = std::max((qp.A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + old_g + qp.A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp3.update(std::nullopt,
             qp.g,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp3.solve();
  pri_res = std::max((qp.A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + qp.g + qp.A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp3.model.g - qp.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp3.results.info.setup_time << " solve time "
            << Qp3.results.info.solve_time << std::endl;

  dense::QP<T> Qp4(dim, n_eq, n_in);
  Qp4.settings.eps_abs = eps_abs;
  Qp4.settings.eps_rel = 0;
  Qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  Qp4.init(qp.H, old_g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp4.solve();
  pri_res = std::max((qp.A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + old_g + qp.A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp4.update(std::nullopt,
             qp.g,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp4.solve();
  pri_res = std::max((qp.A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + qp.g + qp.A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp4.model.g - qp.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp4.results.info.setup_time << " solve time "
            << Qp4.results.info.solve_time << std::endl;

  dense::QP<T> Qp5(dim, n_eq, n_in);
  Qp5.settings.eps_abs = eps_abs;
  Qp5.settings.eps_rel = 0;
  Qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp5.init(qp.H, old_g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
  pri_res = std::max((qp.A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + old_g + qp.A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp5.update(std::nullopt,
             qp.g,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp5.solve();
  pri_res = std::max((qp.A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + qp.g + qp.A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp5.model.g - qp.g).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp5.results.info.setup_time << " solve time "
            << Qp5.results.info.solve_time << std::endl;
}

TEST_CASE("Test A update for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test A update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  auto new_A = test::rand::sparse_matrix_rand_not_compressed<T>(
    n_eq, dim, sparsity_factor);
  Qp.update(std::nullopt,
            std::nullopt,
            new_A,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);
  Qp.solve();
  pri_res = std::max((new_A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + new_A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2(dim, n_eq, n_in);
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp2.update(std::nullopt,
             std::nullopt,
             new_A,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp2.solve();
  pri_res = std::max((new_A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + new_A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp2.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;

  dense::QP<T> Qp3(dim, n_eq, n_in);
  Qp3.settings.eps_abs = eps_abs;
  Qp3.settings.eps_rel = 0;
  Qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp3.solve();
  pri_res = std::max((qp.A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + qp.g + qp.A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp3.update(std::nullopt,
             std::nullopt,
             new_A,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp3.solve();
  pri_res = std::max((new_A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + qp.g + new_A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp3.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp3.results.info.setup_time << " solve time "
            << Qp3.results.info.solve_time << std::endl;

  dense::QP<T> Qp4(dim, n_eq, n_in);
  Qp4.settings.eps_abs = eps_abs;
  Qp4.settings.eps_rel = 0;
  Qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp4.solve();
  pri_res = std::max((qp.A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + qp.g + qp.A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp4.update(std::nullopt,
             std::nullopt,
             new_A,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp4.solve();
  pri_res = std::max((new_A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + qp.g + new_A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp4.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp4.results.info.setup_time << " solve time "
            << Qp4.results.info.solve_time << std::endl;

  dense::QP<T> Qp5(dim, n_eq, n_in);
  Qp5.settings.eps_abs = eps_abs;
  Qp5.settings.eps_rel = 0;
  Qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
  pri_res = std::max((qp.A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + qp.g + qp.A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp5.update(std::nullopt,
             std::nullopt,
             new_A,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt);
  Qp5.solve();
  pri_res = std::max((new_A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + qp.g + new_A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK((Qp5.model.A - new_A).lpNorm<Eigen::Infinity>() <= eps_abs);
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp5.results.info.setup_time << " solve time "
            << Qp5.results.info.solve_time << std::endl;
}

TEST_CASE("Test rho update for different initial guess")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;

  std::cout << "Test rho update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp.update(std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            true,
            T(1.E-7));
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(Qp.results.info.rho == T(1.E-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2(dim, n_eq, n_in);
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Qp2.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp2.update(std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             true,
             T(1.E-7));
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + qp.g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(Qp2.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;

  dense::QP<T> Qp3(dim, n_eq, n_in);
  Qp3.settings.eps_abs = eps_abs;
  Qp3.settings.eps_rel = 0;
  Qp3.settings.initial_guess =
    InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
  Qp3.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp3.solve();
  pri_res = std::max((qp.A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + qp.g + qp.A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp3.update(std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             true,
             T(1.E-7));
  Qp3.solve();
  pri_res = std::max((qp.A * Qp3.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp3.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp3.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp3.results.x + qp.g + qp.A.transpose() * Qp3.results.y +
             qp.C.transpose() * Qp3.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(Qp3.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp3.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp3.results.info.setup_time << " solve time "
            << Qp3.results.info.solve_time << std::endl;

  dense::QP<T> Qp4(dim, n_eq, n_in);
  Qp4.settings.eps_abs = eps_abs;
  Qp4.settings.eps_rel = 0;
  Qp4.settings.initial_guess =
    InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
  Qp4.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp4.solve();
  pri_res = std::max((qp.A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + qp.g + qp.A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp4.update(std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             true,
             T(1.E-7));
  Qp4.solve();
  pri_res = std::max((qp.A * Qp4.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp4.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp4.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp4.results.x + qp.g + qp.A.transpose() * Qp4.results.y +
             qp.C.transpose() * Qp4.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(Qp4.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp4.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp4.results.info.setup_time << " solve time "
            << Qp4.results.info.solve_time << std::endl;

  dense::QP<T> Qp5(dim, n_eq, n_in);
  Qp5.settings.eps_abs = eps_abs;
  Qp5.settings.eps_rel = 0;
  Qp5.settings.initial_guess = InitialGuessStatus::WARM_START;
  Qp5.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp5.solve(Qp3.results.x, Qp3.results.y, Qp3.results.z);
  pri_res = std::max((qp.A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + qp.g + qp.A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  Qp5.update(std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             true,
             T(1.E-7));
  Qp5.solve();
  pri_res = std::max((qp.A * Qp5.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp5.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp5.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp5.results.x + qp.g + qp.A.transpose() * Qp5.results.y +
             qp.C.transpose() * Qp5.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(Qp5.results.info.rho == T(1.e-7));
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp5.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp5.results.info.setup_time << " solve time "
            << Qp5.results.info.solve_time << std::endl;
}

TEST_CASE("Test g update for different warm start with previous result option")
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);
  test::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);

  Qp.settings.eps_abs = eps_abs;
  Qp.settings.eps_rel = 0;
  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;

  std::cout << "Test rho update for different initial guess" << std::endl;
  std::cout << "dirty workspace before any solving: " << Qp.work.dirty
            << std::endl;

  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();

  T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                       (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                        dense::negative_part(qp.C * Qp.results.x - qp.l))
                         .lpNorm<Eigen::Infinity>());
  T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
               qp.C.transpose() * Qp.results.z)
                .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  // a new linear cost slightly modified
  auto g = qp.g * 0.95;

  Qp.update(std::nullopt,
            g,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);
  Qp.solve();
  pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp.results.x + g + qp.A.transpose() * Qp.results.y +
             qp.C.transpose() * Qp.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp.results.info.setup_time << " solve time "
            << Qp.results.info.solve_time << std::endl;

  dense::QP<T> Qp2(dim, n_eq, n_in);
  Qp2.settings.eps_abs = eps_abs;
  Qp2.settings.eps_rel = 0;
  Qp2.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Qp2.init(qp.H, g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp2.solve();
  pri_res = std::max((qp.A * Qp2.results.x - qp.b).lpNorm<Eigen::Infinity>(),
                     (dense::positive_part(qp.C * Qp2.results.x - qp.u) +
                      dense::negative_part(qp.C * Qp2.results.x - qp.l))
                       .lpNorm<Eigen::Infinity>());
  dua_res = (qp.H * Qp2.results.x + g + qp.A.transpose() * Qp2.results.y +
             qp.C.transpose() * Qp2.results.z)
              .lpNorm<Eigen::Infinity>();
  CHECK(dua_res <= eps_abs);
  CHECK(pri_res <= eps_abs);
  std::cout << "--n = " << dim << " n_eq " << n_eq << " n_in " << n_in
            << std::endl;
  std::cout << "; dual residual " << dua_res << "; primal residual " << pri_res
            << std::endl;
  std::cout << "total number of iteration: " << Qp2.results.info.iter
            << std::endl;
  std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time "
            << Qp2.results.info.solve_time << std::endl;
}
