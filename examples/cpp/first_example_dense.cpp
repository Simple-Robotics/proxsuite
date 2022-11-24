//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp> // for c++14
#include <proxsuite/proxqp/dense/dense.hpp>

using namespace proxsuite::proxqp;
using proxsuite::nullopt; // c++17 simply use std::nullopt

int
main()
{
  std::cout
    << "Solve a simple example with inequality constraints using dense ProxQP"
    << std::endl;

  // define the problem
  double eps_abs = 1e-9;
  dense::isize dim = 3, n_eq = 0, n_in = 3;

  // cost H
  Eigen::MatrixXd H = Eigen::MatrixXd(dim, dim);
  H << 13.0, 12.0, -2.0, 12.0, 17.0, 6.0, -2.0, 6.0, 12.0;

  Eigen::VectorXd g = Eigen::VectorXd(dim);
  g << -22.0, -14.5, 13.0;

  // inequality constraints C
  Eigen::MatrixXd C = Eigen::MatrixXd(n_in, dim);
  C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd l = Eigen::VectorXd(n_in);
  l << -1.0, -1.0, -1.0;

  Eigen::VectorXd u = Eigen::VectorXd(n_in);
  u << 1.0, 1.0, 1.0;

  std::cout << "H:\n" << H << std::endl;
  std::cout << "g.T:" << g.transpose() << std::endl;
  std::cout << "C:\n" << C << std::endl;
  std::cout << "l.T:" << l.transpose() << std::endl;
  std::cout << "u.T:" << u.transpose() << std::endl;

  // create qp object and pass some settings
  dense::QP<double> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  qp.settings.verbose = true;

  // initialize qp with matrices describing the problem
  // note: it is also possible to use update here
  qp.init(H, g, nullopt, nullopt, C, l, u);

  qp.solve();

  std::cout << "primal residual: " << qp.results.info.pri_res << std::endl;
  std::cout << "dual residual: " << qp.results.info.dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  return 0;
}