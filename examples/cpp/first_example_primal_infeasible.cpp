//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp> // for c++14
#include <proxsuite/osqp/dense/dense.hpp>

using namespace proxsuite::osqp;
using proxsuite::nullopt;

int
main()
{
  std::cout
    << "Case of primal infeasible problem with box constraints and primal infeasibility solving activated"
    << std::endl;

  // define the problem
  double eps_abs = 1e-9;
  dense::isize dim = 3, n_eq = 3, n_in = 3;

  // cost H
  Eigen::MatrixXd H = Eigen::MatrixXd(dim, dim);
  // H << 13.0, 12.0, -2.0, 12.0, 17.0, 6.0, -2.0, 6.0, 12.0;
  H << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  Eigen::VectorXd g = Eigen::VectorXd(dim);
  g << -1.0, -1.0, -1.0;

  // equality constraints A (A in singular then x is constrained to be equal to -4, -4, -4)
  Eigen::MatrixXd A = Eigen::MatrixXd(n_eq, dim);
  A << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd b = Eigen::VectorXd(n_eq);
  b << -4.0, -4.0, -4.0;

  // inequality constraints C
  Eigen::MatrixXd C = Eigen::MatrixXd(n_in, dim);
  C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd l = Eigen::VectorXd(n_in);
  l << 20.0, 20.0, 20.0; // x should be equal to -1, -1, -1 from Ax = b -> primal infeasibility expected

  Eigen::VectorXd u = Eigen::VectorXd(n_in);
  u << 50.0, 50.0, 50.0;

  Eigen::VectorXd l_box = Eigen::VectorXd(n_in);
  l_box << 2.0, 2.0, 2.0; // x should be equal to -1, -1, -1 from Ax = b -> primal infeasibility expected

  Eigen::VectorXd u_box = Eigen::VectorXd(n_in);
  u_box << 5.0, 5.0, 5.0;

  std::cout << "H:\n" << H << std::endl;
  std::cout << "g.T:" << g.transpose() << std::endl;
  std::cout << "A:\n" << A << std::endl;
  std::cout << "b.T:" << b.transpose() << std::endl;
  std::cout << "C:\n" << C << std::endl;
  std::cout << "l.T:" << l.transpose() << std::endl;
  std::cout << "u.T:" << u.transpose() << std::endl;
  std::cout << "l_box.T:" << l_box.transpose() << std::endl;
  std::cout << "u_box.T:" << u_box.transpose() << std::endl;

  // create qp object and pass some settings
  bool box_constraints = false;
  dense::QP<double> qp(dim, n_eq, n_in, box_constraints, proxsuite::proxqp::DenseBackend::PrimalDualLDLT);

  qp.settings.eps_abs = eps_abs;
  qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  qp.settings.verbose = true;

  // initialize qp with matrices describing the problem
  // note: it is also possible to use update here
  // qp.init(H, g, A, b, C, l, u);
  qp.init(H, g, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt);

  qp.solve();

  std::cout << "unscaled x: " << qp.results.x << std::endl;
  std::cout << "primal residual: " << qp.results.info.pri_res << std::endl;
  std::cout << "dual residual: " << qp.results.info.dua_res << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  return 0;
}