#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <util.hpp>

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

  test::RandomQP<T> qp{ test::random_with_dim_and_neq_and_n_in,
                        dim,
                        n_eq,
                        n_in,
                        sparsity_factor,
                        strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);                  // create the QP object
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // initialize the model
  Qp.solve(); // solve the problem without warm start
  auto x_wm = test::rand::vector_rand<T>(dim);
  auto y_wm = test::rand::vector_rand<T>(n_eq);
  auto z_wm = test::rand::vector_rand<T>(n_in);
  Qp.solve(x_wm, y_wm,
           z_wm); // if you have a warm start, put it here put
}
