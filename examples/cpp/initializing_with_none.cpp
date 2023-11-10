#include <iostream>
#include "proxsuite/proxqp/dense/dense.hpp"
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite::proxqp;
using T = double;
int
main()
{
  dense::isize dim = 10;
  dense::isize n_eq(0);
  dense::isize n_in(0);
  dense::QP<T> qp(dim, n_eq, n_in);
  T strong_convexity_factor(0.1);
  T sparsity_factor(0.15);
  // we generate a qp, so the function used from helpers.hpp is
  // in proxqp namespace. The qp is in dense eigen format and
  // you can control its sparsity ratio and strong convexity factor.
  dense::Model<T> qp_random = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u); // initialization with zero shape matrices
  // it is equivalent to do qp.init(qp_random.H, qp_random.g,
  // nullopt,nullopt,nullopt,nullopt,nullopt);
  qp.solve();
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
}