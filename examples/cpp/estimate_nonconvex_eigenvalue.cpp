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
  // make the QP nonconvex
  dense::Vec<T> diag(dim);
  diag.setOnes();
  qp_random.H.diagonal().array() -=
    diag.array(); // add some nonpositive values dense matrix
  Eigen::SelfAdjointEigenSolver<dense::Mat<T>> es(qp_random.H,
                                                  Eigen::EigenvaluesOnly);
  T minimal_eigenvalue = T(es.eigenvalues().minCoeff());
  // choose the option for estimating this eigenvalue
  qp.settings.find_minimal_H_eigenvalue =
    HessianCostRegularization::EigenRegularization;
  // choose scaling for regularizing default_rho accordingly
  qp.settings.rho_regularization_scaling = T(1.5);
  dense::QP<T> qp(dim, n_eq, n_in); // create the QP object
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u);
  // print the estimates
  std::cout << "ProxQP estimate "
            << qp.results.info.minimal_H_eigenvalue_estimate << std::endl;
  std::cout << "minimal_eigenvalue " << minimal_eigenvalue << std::endl;
  std::cout << "default_rho " << qp.settings.default_rho << std::endl;
}
