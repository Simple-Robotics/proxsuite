#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp> // load the dense solver backend
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

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
  dense::Model<T> qp_random = utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
  // make the QP nonconvex
  dense::Vec<T> diag(dim);
  diag.setOnes();
  qp_random.H.diagonal().array() -=
    2. * diag.array(); // add some nonpositive values dense matrix
  Eigen::SelfAdjointEigenSolver<dense::Mat<T>> es(qp_random.H,
                                                  Eigen::EigenvaluesOnly);
  T minimal_eigenvalue = T(es.eigenvalues().minCoeff());
  // choose scaling for regularizing default_rho accordingly
  dense::QP<T> qp(dim, n_eq, n_in); // create the QP object
  // choose the option for estimating this eigenvalue
  T estimate_minimal_eigen_value =
    dense::estimate_minimal_eigen_value_of_symmetric_matrix(
      qp_random.H, EigenValueEstimateMethodOption::ExactMethod, 1.E-6, 10000);
  bool compute_preconditioner = false;
  // input the estimate for making rho appropriate
  qp.init(qp_random.H,
          qp_random.g,
          qp_random.A,
          qp_random.b,
          qp_random.C,
          qp_random.l,
          qp_random.u,
          compute_preconditioner,
          nullopt,
          nullopt,
          nullopt,
          estimate_minimal_eigen_value);
  // print the estimates
  std::cout << "ProxQP estimate "
            << qp.results.info.minimal_H_eigenvalue_estimate << std::endl;
  std::cout << "minimal_eigenvalue " << minimal_eigenvalue << std::endl;
  std::cout << "default_rho " << qp.settings.default_rho << std::endl;
}
