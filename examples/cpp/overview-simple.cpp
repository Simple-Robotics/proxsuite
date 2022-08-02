#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>// used for generating a random convex Qp

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  // generate a QP problem
  T sparsity_factor = 0.15;
  dense::isize dim = 10;
  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  // we generate a qp, so the function used from helpers.hpp is 
  // in proxqp namespace. The qp is in dense eigen format and 
  // you can control its sparsity ratio and strong convexity factor.
  dense::Model<T> qp = utils::dense_strongly_convex_qp(
                                  dim,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

  // load PROXQP solver with dense backend and solve the problem
  dense::QP<T> Qp(dim, n_eq, n_in);
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
  Qp.solve();
}
