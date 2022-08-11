#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>// used for generating a random convex Qp

using namespace proxsuite;
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
  dense::Model<T> qp = utils::dense_strongly_convex_qp(
                                  dim,
                                  n_eq,
                                  n_in,
                                  sparsity_factor,
                                  strong_convexity_factor);

  dense::QP<T> Qp(dim, n_eq, n_in);          // create the QP object
  Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // initialize the model
  Qp.solve();                                        // solve the problem
  // re update the linear cost taking previous result
  Qp.settings.initial_guess =
    InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  // it takes effect at the update because it is set before
  // (the workspace is not erased at the update method, hence
  // the previous factorization is kept)
  // a new linear cost slightly modified
  auto g = qp.g * 0.95;
  Qp.update(std::nullopt,
            g,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);
  Qp.solve();
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << Qp.results.x << std::endl;
  std::cout << "optimal y: " << Qp.results.y << std::endl;
  std::cout << "optimal z: " << Qp.results.z << std::endl;
}
