#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  dense::isize dim = 10;
  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
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
          qp_random.u);           // initialize the model
  qp.settings.eps_abs = T(1.E-9); // set accuracy threshold to 1.e-9
  qp.settings.verbose = true;     // print some intermediary results
  qp.solve();                     // solve the problem with previous settings
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
}
