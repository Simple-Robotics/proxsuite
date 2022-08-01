#include <qp/dense/dense.hpp>    // load the dense solver backend
#include <test/include/util.hpp> // used for generating a random convex Qp

using namespace qp;
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
  Qp<T> qp{ random_with_dim_and_neq_and_n_in, dim, n_eq, n_in, sparsity_factor,
            strong_convexity_factor };

  dense::QP<T> Qp(dim, n_eq, n_in);                  // create the QP object
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // initialize the model
}
