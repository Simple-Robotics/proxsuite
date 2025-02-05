#include <iostream>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite::osqp;
using T = double;

int
main()
{
  // generate a QP problem
  T sparsity_factor = 0.15;
  proxsuite::proxqp::dense::isize dim = 20;
  proxsuite::proxqp::dense::isize n_eq(dim / 4);
  proxsuite::proxqp::dense::isize n_in(0);
  T strong_convexity_factor(1.e-2);
  // we generate a qp, so the function used from helpers.hpp is
  // in proxqp namespace. The qp is in dense eigen format and
  // you can control its sparsity ratio and strong convexity factor.
  proxsuite::proxqp::dense::Model<T> qp_random = proxsuite::proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  // load PROXQP solver with dense backend and solve the problem
  dense::QP<T> qp(dim, n_eq, n_in);
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          proxsuite::nullopt,
          proxsuite::nullopt,
          proxsuite::nullopt);
  qp.solve();
  // print an optimal solution x,y and z
  // std::cout << "optimal x: " << qp.results.x << std::endl;
  // std::cout << "optimal y: " << qp.results.y << std::endl;
  // std::cout << "optimal z: " << qp.results.z << std::endl;
  std::cout << "Primal residual: " << qp.results.info.pri_res << std::endl;
  std::cout << "Dual residual: " << qp.results.info.dua_res << std::endl;
  std::cout << "Duality gap: " << qp.results.info.duality_gap << std::endl;
  std::cout << "Setup time: " << qp.results.info.setup_time << " Solve time: " << qp.results.info.solve_time << std::endl;
}
