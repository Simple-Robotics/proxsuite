//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp> // for c++14
#include <proxsuite/osqp/dense/dense.hpp>

using namespace proxsuite::osqp;
using proxsuite::nullopt; // c++17 simply use std::nullopt

int
main()
{
  std::cout
    << "Solve a simple example with equality constraints using dense OSQP"
    << std::endl;

  // define the problem
  double eps_abs = 1e-9;
  dense::isize dim = 3, n_eq = 3, n_in = 0;

  // cost H
  Eigen::MatrixXd H = Eigen::MatrixXd(dim, dim);
  H << 13.0, 12.0, -2.0, 12.0, 17.0, 6.0, -2.0, 6.0, 12.0;

  Eigen::VectorXd g = Eigen::VectorXd(dim);
  g << -22.0, -14.5, 13.0;

  // equality constraints A
  Eigen::MatrixXd A = Eigen::MatrixXd(n_eq, dim);
  A << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd b = Eigen::VectorXd(n_eq);
  b << -1.0, -1.0, -1.0;

  std::cout << "H:\n" << H << std::endl;
  std::cout << "g.T:" << g.transpose() << std::endl;
  std::cout << "A:\n" << A << std::endl;
  std::cout << "b.T:" << b.transpose() << std::endl;

  // create qp object and pass some settings
  dense::QP<double> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.initial_guess =
    proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  qp.settings.verbose = true;

  // Specific values for OSQP
  qp.settings.default_mu_eq = 1.e-2;
  qp.settings.default_mu_in = 1.e1;
  qp.settings.eps_rel = 1.e-4;
  qp.settings.check_duality_gap = false;
  qp.settings.eps_duality_gap_abs = 1.e-3;
  qp.settings.eps_duality_gap_rel = 1.e-3;

  // initialize qp with matrices describing the problem
  // note: it is also possible to use update here
  qp.init(H, g, A, b, nullopt, nullopt, nullopt);

  qp.solve();

  std::cout << "primal residual: " << qp.results.info.pri_res << std::endl;
  std::cout << "dual residual: " << qp.results.info.dua_res << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  return 0;
}