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

  dense::QP<T> qp(dim, n_eq, n_in); // create the QP object
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u); // initialize the model
  qp.solve();           // solve the problem without warm start
  auto x_wm = utils::rand::vector_rand<T>(dim);
  auto y_wm = utils::rand::vector_rand<T>(n_eq);
  auto z_wm = utils::rand::vector_rand<T>(n_in);
  qp.solve(x_wm, y_wm,
           z_wm); // if you have a warm start, put it here
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;

  // Another example if you have box constraints (for the dense backend only for
  // the moment)
  dense::QP<T> qp2(dim, n_eq, n_in, true); // create the QP object
  // some trivial boxes
  dense::Vec<T> u_box(dim);
  dense::Vec<T> l_box(dim);
  u_box.setZero();
  l_box.setZero();
  u_box.array() += 1.E10;
  l_box.array() -= 1.E10;
  qp2.init(qp_random.H,
           qp_random.g,
           qp_random.A,
           qp_random.b,
           qp_random.C,
           qp_random.l,
           qp_random.u,
           l_box,
           u_box); // initialize the model with the boxes
  qp2.solve();     // solve the problem
  // An important note regarding the inequality multipliers
  // auto z_ineq = qp.results.z.head(n_in); contains the multiplier associated
  // to qp_random.C z_box = qp.results.z.tail(dim); the last dim elements
  // correspond to multiplier associated to the box constraints
}
