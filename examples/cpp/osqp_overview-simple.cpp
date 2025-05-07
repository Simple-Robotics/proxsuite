#include <iostream>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite::osqp;
using T = double;

namespace pp = proxsuite::proxqp;
namespace ppd = proxsuite::proxqp::dense;
namespace pod = proxsuite::osqp::dense;

int
main()
{
  T sparsity_factor = 0.15;
  ppd::isize dim = 10;
  ppd::isize n_eq(dim / 4);
  ppd::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);

  ppd::Model<T> qp_random = pp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  pod::QP<T> qp(dim, n_eq, n_in);
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u);
  qp.solve();

  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
}
