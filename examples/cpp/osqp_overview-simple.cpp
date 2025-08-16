#include <iostream>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/common/utils/random_qp_problems.hpp> // used for generating a random convex qp

using T = double;
using namespace proxsuite;
using proxsuite::common::isize;

int
main()
{
  // generate a QP problem
  T sparsity_factor = 0.15;
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);

  common::dense::Model<T> qp_random = common::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  // load OSQP solver with dense backend and solve the problem
  osqp::dense::QP<T> qp(dim, n_eq, n_in);
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u);
  qp.solve();
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
}
