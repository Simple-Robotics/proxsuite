//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp> // for c++14
// #include <proxsuite/proxqp/dense/dense.hpp>

#include <iostream>
#include <Eigen/Cholesky>
#include <proxsuite/osqp/dense/dense.hpp>
#include <proxsuite/linalg/veg/util/dbg.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

// using namespace proxsuite::proxqp;
using namespace proxsuite::osqp;
using namespace proxsuite;
using proxsuite::nullopt;

// int
// main()
// {
//   std::cout
//     << "Case of primal infeasible problem with box constraints and primal
//     infeasibility solving activated"
//     << std::endl;

//   // define the problem
//   double eps_abs = 1e-9;
//   dense::isize dim = 3, n_eq = 3, n_in = 3;

//   // cost H
//   Eigen::MatrixXd H = Eigen::MatrixXd(dim, dim);
//   H << 13.0, 12.0, -2.0, 12.0, 17.0, 6.0, -2.0, 6.0, 12.0;

//   Eigen::VectorXd g = Eigen::VectorXd(dim);
//   g << -1.0, -1.0, -1.0;

//   Eigen::MatrixXd A = Eigen::MatrixXd(n_eq, dim);
//   A << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

//   Eigen::VectorXd b = Eigen::VectorXd(n_eq);
//   b << 15.0, 15.0, 15.0;

//   // inequality constraints C
//   Eigen::MatrixXd C = Eigen::MatrixXd(n_in, dim);
//   C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

//   Eigen::VectorXd l = Eigen::VectorXd(n_in);
//   l << -1.0, -1.0, -1.0;

//   Eigen::VectorXd u = Eigen::VectorXd(n_in);
//   u << 1.0, 1.0, 1.0;

//   // Eigen::VectorXd l_box = Eigen::VectorXd(dim);
//   // l_box << -10.0, -10.0, -10.0;

//   // Eigen::VectorXd u_box = Eigen::VectorXd(dim);
//   // u_box << 10.0, 10.0, 10.0;

//   std::cout << "H:\n" << H << std::endl;
//   std::cout << "g.T:" << g.transpose() << std::endl;
//   std::cout << "A:\n" << A << std::endl;
//   std::cout << "b.T:" << b.transpose() << std::endl;
//   std::cout << "C:\n" << C << std::endl;
//   std::cout << "l.T:" << l.transpose() << std::endl;
//   std::cout << "u.T:" << u.transpose() << std::endl;
//   // std::cout << "l_box.T:" << l_box.transpose() << std::endl;
//   // std::cout << "u_box.T:" << u_box.transpose() << std::endl;

//   // create qp object and pass some settings
//   // bool box_constraints = true;
//   bool box_constraints = false;
//   dense::QP<double> qp(dim, n_eq, n_in, box_constraints,
//   proxsuite::proxqp::DenseBackend::PrimalDualLDLT);

//   qp.settings.eps_abs = eps_abs;
//   qp.settings.initial_guess =
//   proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
//   qp.settings.verbose = true;

//   // initialize qp with matrices describing the problem
//   // note: it is also possible to use update here
//   // qp.init(H, g, A, b, C, l, u, l_box, u_box);
//   qp.init(H, g, A, b, C, l, u);

//   qp.settings.primal_infeasibility_solving = true;

//   qp.solve();

//   std::cout << "unscaled x: " << qp.results.x << std::endl;
//   std::cout << "primal residual: " << qp.results.info.pri_res << std::endl;
//   std::cout << "dual residual: " << qp.results.info.dua_res << std::endl;
//   std::cout << "setup timing " << qp.results.info.setup_time << " solve time
//   "
//             << qp.results.info.solve_time << std::endl;
//   return 0;
// }

int
main()
{
  std::cout << "---testing linear problem with equality constraints and "
               "increasing dimension using wrapper API---"
            << std::endl;
  double sparsity_factor = 0.15;
  double eps_abs = 1e-9;
  proxqp::utils::rand::set_seed(1);
  for (proxqp::isize dim = 10; dim < 1000; dim += 100) {

    proxqp::isize n_eq(dim / 2);
    proxqp::isize n_in(0);
    double strong_convexity_factor(1.e-2);
    proxqp::dense::Model<double> qp_random =
      proxqp::utils::dense_strongly_convex_qp(
        dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp_random.H.setZero();
    auto y_sol = proxqp::utils::rand::vector_rand<double>(
      n_eq); // make sure the LP is bounded within the feasible set
    qp_random.g = -qp_random.A.transpose() * y_sol;

    // osqp::dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
    bool box_constraints = false;
    osqp::dense::QP<double> qp{
      dim, n_eq, n_in, box_constraints, proxqp::DenseBackend::PrimalLDLT
    }; // creating QP object
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qp.solve();

    double pri_res = std::max(
      (qp_random.A * qp.results.x - qp_random.b).lpNorm<Eigen::Infinity>(),
      (helpers::positive_part(qp_random.C * qp.results.x - qp_random.u) +
       helpers::negative_part(qp_random.C * qp.results.x - qp_random.l))
        .lpNorm<Eigen::Infinity>());
    double dua_res = (qp_random.H * qp.results.x + qp_random.g +
                      qp_random.A.transpose() * qp.results.y +
                      qp_random.C.transpose() * qp.results.z)
                       .lpNorm<Eigen::Infinity>();

    std::cout << "------using wrapper API solving qp with dim: " << dim
              << " neq: " << n_eq << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: " << dua_res << std::endl;
    std::cout << "total number of iteration: " << qp.results.info.iter
              << std::endl;
  }
  return 0;
}
