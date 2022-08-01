#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <util.hpp> // used for generating a random convex Qp

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

  dense::QP<T> Qp(
    dim, n_eq, n_in); // create the QP
                      // initialize the model, along with another rho parameter
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l, true, /*rho*/ 1.e-7);
  // in c++ you must follow the order speficied in the API for the parameters
  // if you don't want to change one parameter (here compute_preconditioner),
  // just let it be std::nullopt
}
