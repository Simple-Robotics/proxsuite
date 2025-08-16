#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/common/utils/random_qp_problems.hpp> // used for generating a random convex qp

using T = double;
using namespace proxsuite;
using proxsuite::common::isize;

int
main()
{
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  // generate a random qp
  T sparsity_factor(0.15);
  T strong_convexity_factor(1.e-2);
  common::dense::Model<T> qp_random = common::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  proxqp::dense::QP<T> qp(dim, n_eq, n_in); // create the QP object
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u); // initialize the model
}
