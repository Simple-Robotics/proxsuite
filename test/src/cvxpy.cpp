//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <proxsuite/proxqp/dense/dense.hpp>

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

template<typename T, proxqp::Layout L>
using Mat =
  Eigen::Matrix<T,
                Eigen::Dynamic,
                Eigen::Dynamic,
                (L == proxqp::colmajor) ? Eigen::ColMajor : Eigen::RowMajor>;
template<typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

DOCTEST_TEST_CASE("3 dim test case from cvxpy, check feasibility")
{

  std::cout << "---3 dim test case from cvxpy, check feasibility " << std::endl;
  T eps_abs = T(1e-9);
  dense::isize dim = 3;

  Mat<T, colmajor> H = Mat<T, colmajor>(dim, dim);
  H << 13.0, 12.0, -2.0, 12.0, 17.0, 6.0, -2.0, 6.0, 12.0;

  Vec<T> g = Vec<T>(dim);
  g << -22.0, -14.5, 13.0;

  Mat<T, colmajor> C = Mat<T, colmajor>(dim, dim);
  C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  Vec<T> l = Vec<T>(dim);
  l << -1.0, -1.0, -1.0;

  Vec<T> u = Vec<T>(dim);
  u << 1.0, 1.0, 1.0;
  Results<T> results = dense::solve<T>(
    H, g, nullopt, nullopt, C, l, u, nullopt, nullopt, nullopt, eps_abs, 0);

  T pri_res = (helpers::positive_part(C * results.x - u) +
               helpers::negative_part(C * results.x - l))
                .lpNorm<Eigen::Infinity>();
  T dua_res =
    (H * results.x + g + C.transpose() * results.z).lpNorm<Eigen::Infinity>();
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("simple test case from cvxpy, check feasibility")
{

  std::cout << "---simple test case from cvxpy, check feasibility "
            << std::endl;
  T eps_abs = T(1e-8);
  dense::isize dim = 1;

  Mat<T, colmajor> H = Mat<T, colmajor>(dim, dim);
  H << 20.0;

  Vec<T> g = Vec<T>(dim);
  g << -10.0;

  Mat<T, colmajor> C = Mat<T, colmajor>(dim, dim);
  C << 1.0;

  Vec<T> l = Vec<T>(dim);
  l << 0.0;

  Vec<T> u = Vec<T>(dim);
  u << 1.0;
  Results<T> results = dense::solve<T>(
    H, g, nullopt, nullopt, C, l, u, nullopt, nullopt, nullopt, eps_abs, 0);

  T pri_res = (helpers::positive_part(C * results.x - u) +
               helpers::negative_part(C * results.x - l))
                .lpNorm<Eigen::Infinity>();
  T dua_res =
    (H * results.x + g + C.transpose() * results.z).lpNorm<Eigen::Infinity>();
  T x_sol = 0.5;

  DOCTEST_CHECK((x_sol - results.x.coeff(0, 0)) <= eps_abs);
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << results.info.iter << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

DOCTEST_TEST_CASE("simple test case from cvxpy, init with solution, check that "
                  "solver stays there")
{

  std::cout << "---simple test case from cvxpy, init with solution, check that "
               "solver stays there"
            << std::endl;
  T eps_abs = T(1e-4);
  dense::isize dim = 1;

  Mat<T, colmajor> H = Mat<T, colmajor>(dim, dim);
  H << 20.0;

  Vec<T> g = Vec<T>(dim);
  g << -10.0;

  Mat<T, colmajor> C = Mat<T, colmajor>(dim, dim);
  C << 1.0;

  Vec<T> l = Vec<T>(dim);
  l << 0.0;

  Vec<T> u = Vec<T>(dim);
  u << 1.0;

  T x_sol = 0.5;

  proxqp::isize n_in(1);
  proxqp::isize n_eq(0);
  proxqp::dense::QP<T> qp{ dim, n_eq, n_in };
  qp.settings.eps_abs = eps_abs;

  qp.init(H, g, nullopt, nullopt, C, u, l);

  dense::Vec<T> x = dense::Vec<T>(dim);
  dense::Vec<T> z = dense::Vec<T>(n_in);
  x << 0.5;
  z << 0.0;
  qp.solve(x, nullopt, z);

  T pri_res = (helpers::positive_part(C * qp.results.x - u) +
               helpers::negative_part(C * qp.results.x - l))
                .lpNorm<Eigen::Infinity>();
  T dua_res = (H * qp.results.x + g + C.transpose() * qp.results.z)
                .lpNorm<Eigen::Infinity>();

  DOCTEST_CHECK(qp.results.info.iter <= 0);
  DOCTEST_CHECK((x_sol - qp.results.x.coeff(0, 0)) <= eps_abs);
  DOCTEST_CHECK(pri_res <= eps_abs);
  DOCTEST_CHECK(dua_res <= eps_abs);

  std::cout << "primal residual: " << pri_res << std::endl;
  std::cout << "dual residual: " << dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
}
