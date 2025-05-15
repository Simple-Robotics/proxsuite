#include <iostream>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

#include <Eigen/Core>

using namespace proxsuite::osqp;
using T = double;

namespace pp = proxsuite::proxqp;
namespace ppd = proxsuite::proxqp::dense;
namespace pod = proxsuite::osqp::dense;

int
main()
{
  T sparsity_factor = 0.15;
  ppd::isize dim = 110;
  ppd::isize n_eq(dim / 4);
  ppd::isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);

  ppd::Model<T> qp_random = pp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  // // Proxqp
  // ppd::QP<T> qp_proxqp(dim, n_eq, n_in);

  // qp_proxqp.init(qp_random.H,
  //         qp_random.g,
  //         qp_random.A,
  //         qp_random.b,
  //         qp_random.C,
  //         qp_random.l,
  //         qp_random.u);
  // qp_proxqp.solve();

  // // Osqp without update
  // pod::QP<T> qp_osqp(dim, n_eq, n_in);

  // qp_osqp.settings.default_mu_eq = T(1.E-2);
  // qp_osqp.settings.default_mu_in = T(1.E1);

  // qp_osqp.init(qp_random.H,
  //         qp_random.g,
  //         qp_random.A,
  //         qp_random.b,
  //         qp_random.C,
  //         qp_random.l,
  //         qp_random.u);
  // qp_osqp.solve();

  // qp_osqp_1 with mu_update
  // pod::QP<T> qp_osqp_1(dim, n_eq, n_in);

  // qp_osqp_1.settings.default_mu_eq = T(1.E-2);
  // qp_osqp_1.settings.default_mu_in = T(1.E1);

  // qp_osqp_1.settings.update_mu = true;

  // qp_osqp_1.init(qp_random.H,
  //         qp_random.g,
  //         qp_random.A,
  //         qp_random.b,
  //         qp_random.C,
  //         qp_random.l,
  //         qp_random.u);
  // qp_osqp_1.solve();
}
