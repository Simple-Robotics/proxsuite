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

  dense::QP<T> Qp(
    dim, n_eq, n_in); // create the QP
                      // initialize the model, along with another rho parameter
  Qp.settings.initial_guess = = proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  Qp.init(qp.H,
          qp.g,
          qp.A,
          qp.b,
          qp.C,
          qp.u,
          qp.l,
          true,
          /*rho*/ 1.e-7,
          /*mu_eq*/ 1.e-4);
  // Initializing rho sets in practive Qp.settings.default_rho value,
  // hence, after each solve or update method, the Qp.results.info.rho value
  // will be reset to Qp.settings.default_rho value.
  Qp.solve();
  // So if we redo a solve, Qp.settings.default_rho value = 1.e-7, hence
  // Qp.results.info.rho restarts at 1.e-7 The same occurs for mu_eq.
  Qp.solve();
  // There might be a different result with WARM_START_WITH_PREVIOUS_RESULT
  // initial guess option, as by construction, it reuses the last proximal step
  // sizes of the last solving method.
}
