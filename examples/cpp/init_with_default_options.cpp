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

  dense::QP<T> qp(
    dim, n_eq, n_in); // create the QP
                      // initialize the model, along with another rho parameter
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          true,
          /*rho*/ 1.e-7,
          /*mu_eq*/ 1.e-4);
  // Initializing rho sets in practive qp.settings.default_rho value,
  // hence, after each solve or update method, the qp.results.info.rho value
  // will be reset to qp.settings.default_rho value.
  qp.solve();
  // So if we redo a solve, qp.settings.default_rho value = 1.e-7, hence
  // qp.results.info.rho restarts at 1.e-7 The same occurs for mu_eq.
  qp.solve();
  // There might be a different result with WARM_START_WITH_PREVIOUS_RESULT
  // initial guess option, as by construction, it reuses the last proximal step
  // sizes of the last solving method.
}
