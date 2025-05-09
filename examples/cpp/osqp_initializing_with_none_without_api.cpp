#include <iostream>
#include "proxsuite/helpers/optional.hpp"
#include "proxsuite/osqp/dense/dense.hpp"
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite;
using T = double;

namespace pp = proxsuite::proxqp;
namespace ppd = proxsuite::proxqp::dense;
namespace pod = proxsuite::osqp::dense;

int
main()
{
  ppd::isize dim = 10;
  ppd::isize n_eq(0);
  ppd::isize n_in(0);
  T strong_convexity_factor(0.1);
  T sparsity_factor(0.15);
  // we generate a qp, so the function used from helpers.hpp is
  // in proxqp namespace. The qp is in dense eigen format and
  // you can control its sparsity ratio and strong convexity factor.
  ppd::Model<T> qp_random = pp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  T eps_abs(1.E-5);
  T eps_rel(0);
  pp::Results<T> results = pod::solve<T>(qp_random.H,
                                         qp_random.g,
                                         qp_random.A,
                                         qp_random.b,
                                         qp_random.C,
                                         qp_random.l,
                                         qp_random.u,
                                         nullopt,
                                         nullopt,
                                         nullopt,
                                         eps_abs,
                                         eps_rel,
                                         nullopt,
                                         T(1.E-2),
                                         T(1.E-1));

  // initialization with zero shape matrices
  // it is equivalent to do dense::solve<T>(qp_random.H, qp_random.g,
  // nullopt,nullopt,nullopt,nullopt,nullopt);
  //  print an optimal solution x,y and z
  std::cout << "optimal x: " << results.x << std::endl;
  std::cout << "optimal y: " << results.y << std::endl;
  std::cout << "optimal z: " << results.z << std::endl;
}