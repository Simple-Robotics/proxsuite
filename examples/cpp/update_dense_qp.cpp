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
  qp.solve();           // solve the problem
                        // a new qp problem
  dense::Model<T> qp2 = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  // re update the model
  qp.update(qp2.H, qp2.g, qp2.A, qp2.b, qp2.C, qp2.l, qp2.u);
  // solve it
  qp.solve();
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
  // if you have boxes (dense backend only) you proceed the same way
  dense::QP<T> qp_box(dim, n_eq, n_in, true); // create the QP object
  dense::Vec<T> u_box(dim);
  dense::Vec<T> l_box(dim);
  u_box.setZero();
  l_box.setZero();
  u_box.array() += 1.E10;
  l_box.array() -= 1.E10;
  qp_box.init(qp_random.H,
              qp_random.g,
              qp_random.A,
              qp_random.b,
              qp_random.C,
              qp_random.l,
              qp_random.u,
              l_box,
              u_box);
  qp_box.solve();
  u_box.array() += 1.E1;
  l_box.array() -= 1.E1;
  qp_box.update(qp2.H, qp2.g, qp2.A, qp2.b, qp2.C, qp2.l, qp2.u, l_box, u_box);
  qp_box.solve();
}
