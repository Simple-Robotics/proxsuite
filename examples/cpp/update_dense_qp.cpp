#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex Qp

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
  dense::Model<T> qp = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  dense::QP<T> Qp(dim, n_eq, n_in);                  // create the QP object
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // initialize the model
  Qp.solve();                                        // solve the problem
                                                     // a new Qp problem
  dense::Model<T> qp2 = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  // re update the model
  Qp.update(qp2.H, qp2.g, qp2.A, qp2.b, qp2.C, qp2.u, qp2.l);
  // solve it
  Qp.solve();
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << Qp.results.x << std::endl;
  std::cout << "optimal y: " << Qp.results.y << std::endl;
  std::cout << "optimal z: " << Qp.results.z << std::endl;
}
