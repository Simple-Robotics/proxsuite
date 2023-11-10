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
  T strong_convexity_factor(0.1);
  T sparsity_factor(0.15);
  // we generate a qp, so the function used from helpers.hpp is
  // in proxqp namespace. The qp is in dense eigen format and
  // you can control its sparsity ratio and strong convexity factor.
  dense::Model<T> qp_random = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  Results<T> results =
    dense::solve<T>(qp_random.H,
                    qp_random.g,
                    qp_random.A,
                    qp_random.b,
                    qp_random.C,
                    qp_random.l,
                    qp_random.u); // initialization with zero shape matrices
  // it is equivalent to do dense::solve<T>(qp_random.H, qp_random.g,
  // nullopt,nullopt,nullopt,nullopt,nullopt);
  //  print an optimal solution x,y and z
  std::cout << "optimal x: " << results.x << std::endl;
  std::cout << "optimal y: " << results.y << std::endl;
  std::cout << "optimal z: " << results.z << std::endl;
}