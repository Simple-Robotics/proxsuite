//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <proxsuite/osqp/dense/dense.hpp>

using T = double;
using namespace proxsuite;

namespace pp = proxsuite::proxqp;
namespace ppd = proxsuite::proxqp::dense;
namespace pod = proxsuite::osqp::dense;

template<typename T, proxqp::Layout L>
using Mat =
  Eigen::Matrix<T,
                Eigen::Dynamic,
                Eigen::Dynamic,
                (L == proxqp::colmajor) ? Eigen::ColMajor : Eigen::RowMajor>;
template<typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// // 1
// DOCTEST_TEST_CASE("3 dim test case from cvxpy, check feasibility")
// {

//   std::cout << "---3 dim test case from cvxpy, check feasibility " <<
//   std::endl; T eps_abs = T(1e-9); ppd::isize dim = 3;

//   Mat<T, pp::colmajor> H = Mat<T, pp::colmajor>(dim, dim);
//   H << 13.0, 12.0, -2.0, 12.0, 17.0, 6.0, -2.0, 6.0, 12.0;

//   Vec<T> g = Vec<T>(dim);
//   g << -22.0, -14.5, 13.0;

//   Mat<T, pp::colmajor> C = Mat<T, pp::colmajor>(dim, dim);
//   C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

//   Vec<T> l = Vec<T>(dim);
//   l << -1.0, -1.0, -1.0;

//   Vec<T> u = Vec<T>(dim);
//   u << 1.0, 1.0, 1.0;
//   // pp::Results<T> results = pod::solve<T>(
//   //   H, g, nullopt, nullopt, C, l, u, nullopt, nullopt, nullopt, eps_abs,
//   0); pp::Results<T> results = pod::solve<T>(H,
//                                          g,
//                                          nullopt,
//                                          nullopt,
//                                          C,
//                                          l,
//                                          u,
//                                          nullopt,
//                                          nullopt,
//                                          nullopt,
//                                          eps_abs,
//                                          0,
//                                          T(1.E-6),
//                                          T(1.E-2),
//                                          T(1.E1));

//   T pri_res = (helpers::positive_part(C * results.x - u) +
//                helpers::negative_part(C * results.x - l))
//                 .lpNorm<Eigen::Infinity>();
//   T dua_res =
//     (H * results.x + g + C.transpose() *
//     results.z).lpNorm<Eigen::Infinity>();
//   DOCTEST_CHECK(pri_res <= eps_abs);
//   DOCTEST_CHECK(dua_res <= eps_abs);

//   std::cout << "primal residual: " << pri_res << std::endl;
//   std::cout << "dual residual: " << dua_res << std::endl;
//   std::cout << "total number of iteration (admm): " << results.info.iter_ext
//             << std::endl;
//   std::cout << "setup timing " << results.info.setup_time << " solve time "
//             << results.info.solve_time << std::endl;
// }

// 2
DOCTEST_TEST_CASE("simple test case from cvxpy, check feasibility")
{

  std::cout << "---simple test case from cvxpy, check feasibility "
            << std::endl;
  T eps_abs = T(1e-8);
  // Solution polishing
  // In that case, the polishing algorithm does not find any upper or lower
  // constraint. Then the algorithm as described in the solver's paper would
  // stop at a low precision (default 1e-3) and the test would fail. Original
  // feature in this implementation (C): Introduction of the option resume_admm
  // that consists in return to the ADMM iterations in case of polishing failing
  // or no active sets founds. In that case, the QPs are directly solved, if
  // possible, without the polishing. Note: Option set to true by default, as it
  // allows the solver to pass the tests and problems in benchmarks, while keep
  // providing relevant information (from polish status).
  ppd::isize dim = 1;

  Mat<T, pp::colmajor> H = Mat<T, pp::colmajor>(dim, dim);
  H << 20.0;

  Vec<T> g = Vec<T>(dim);
  g << -10.0;

  Mat<T, pp::colmajor> C = Mat<T, pp::colmajor>(dim, dim);
  C << 1.0;

  Vec<T> l = Vec<T>(dim);
  l << 0.0;

  Vec<T> u = Vec<T>(dim);
  u << 1.0;
  // pp::Results<T> results = pod::solve<T>(
  //   H, g, nullopt, nullopt, C, l, u, nullopt, nullopt, nullopt, eps_abs, 0);
  pp::Results<T> results = pod::solve<T>(H,
                                         g,
                                         nullopt,
                                         nullopt,
                                         C,
                                         l,
                                         u,
                                         nullopt,
                                         nullopt,
                                         nullopt,
                                         eps_abs,
                                         0,
                                         T(1.E-6),
                                         T(1.E-2),
                                         T(1.E1));

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
  std::cout << "total number of iteration (admm): " << results.info.iter_ext
            << std::endl;
  std::cout << "setup timing " << results.info.setup_time << " solve time "
            << results.info.solve_time << std::endl;
}

// // 3
// DOCTEST_TEST_CASE("simple test case from cvxpy, init with solution, check
// that "
//                   "solver stays there")
// {

//   std::cout << "---simple test case from cvxpy, init with solution, check
//   that "
//                "solver stays there"
//             << std::endl;
//   T eps_abs = T(1e-4);
//   ppd::isize dim = 1;

//   Mat<T, pp::colmajor> H = Mat<T, pp::colmajor>(dim, dim);
//   H << 20.0;

//   Vec<T> g = Vec<T>(dim);
//   g << -10.0;

//   Mat<T, pp::colmajor> C = Mat<T, pp::colmajor>(dim, dim);
//   C << 1.0;

//   Vec<T> l = Vec<T>(dim);
//   l << 0.0;

//   Vec<T> u = Vec<T>(dim);
//   u << 1.0;

//   T x_sol = 0.5;

//   proxqp::isize n_in(1);
//   proxqp::isize n_eq(0);
//   pod::QP<T> qp{ dim, n_eq, n_in };
//   qp.settings.eps_abs = eps_abs;

//   qp.settings.default_mu_eq = T(1.E-2);
//   qp.settings.default_mu_in = T(1.E1);

//   qp.init(H, g, nullopt, nullopt, C, u, l);

//   ppd::Vec<T> x = ppd::Vec<T>(dim);
//   ppd::Vec<T> z = ppd::Vec<T>(n_in);
//   x << 0.5;
//   z << 0.0;
//   qp.solve(x, nullopt, z);

//   T pri_res = (helpers::positive_part(C * qp.results.x - u) +
//                helpers::negative_part(C * qp.results.x - l))
//                 .lpNorm<Eigen::Infinity>();
//   T dua_res = (H * qp.results.x + g + C.transpose() * qp.results.z)
//                 .lpNorm<Eigen::Infinity>();

//   DOCTEST_CHECK(qp.results.info.iter_ext <= 0);
//   DOCTEST_CHECK((x_sol - qp.results.x.coeff(0, 0)) <= eps_abs);
//   DOCTEST_CHECK(pri_res <= eps_abs);
//   DOCTEST_CHECK(dua_res <= eps_abs);

//   std::cout << "primal residual: " << pri_res << std::endl;
//   std::cout << "dual residual: " << dua_res << std::endl;
//   std::cout << "number of iterations (admm): " << qp.results.info.iter_ext
//             << std::endl;
//   if (qp.settings.polish) {
//   std::cout << "number of iterations (polishing): " <<
//   qp.settings.polish_refine_iter
//             << std::endl;
//   }
//   std::cout << "setup timing " << qp.results.info.setup_time << " solve time
//   "
//             << qp.results.info.solve_time << std::endl;
// }