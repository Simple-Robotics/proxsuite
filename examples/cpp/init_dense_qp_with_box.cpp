#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  // generate a random qp
  T sparsity_factor(0.15);
  T strong_convexity_factor(1.e-2);
  dense::Model<T> qp_random = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  // specify some trivial box constraints
  dense::Vec<T> u_box(dim);
  dense::Vec<T> l_box(dim);
  u_box.setZero();
  l_box.setZero();
  u_box.array() += 1.E10;
  l_box.array() -= 1.E10;

  dense::QP<T> qp(dim, n_eq, n_in, true); // create the QP object
  // and specify with true you take into account box constraints
  // in the model
  // you can check that there are constraints with method is_box_constrained
  std::cout << "the qp is box constrained : " << qp.is_box_constrained()
            << std::endl;
  // init the model
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          l_box,
          u_box); // initialize the model
}
