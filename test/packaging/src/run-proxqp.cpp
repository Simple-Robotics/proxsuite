#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/common/utils/random_qp_problems.hpp> // used for generating a random convex Qp

using namespace proxsuite;
using T = double;

int
main()
{
  common::isize dim = 10;
  common::isize n_eq(dim / 4);
  common::isize n_in(dim / 4);
  // generate a random qp
  T sparsity_factor(0.15);
  T strong_convexity_factor(1.e-2);

  common::dense::Model<T> qp = common::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  proxqp::dense::QP<T> Qp(dim, n_eq, n_in);          // create the QP object
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.l, qp.u); // initialize the model
  Qp.solve(); // solve the problem without warm start

  auto x_wm = common::utils::rand::vector_rand<T>(dim);
  auto y_wm = common::utils::rand::vector_rand<T>(n_eq);
  auto z_wm = common::utils::rand::vector_rand<T>(n_in);
  Qp.solve(x_wm, y_wm,
           z_wm); // if you have a warm start, put it here

  // print an optimal solution x,y and z
  std::cout << "optimal x: " << Qp.results.x << std::endl;
  std::cout << "optimal y: " << Qp.results.y << std::endl;
  std::cout << "optimal z: " << Qp.results.z << std::endl;

  return 0;
}
