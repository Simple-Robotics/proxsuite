//
// Copyright (c) 2022 INRIA
//
#include <iostream>
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp> // for c++14
#include <proxsuite/proxqp/dense/dense.hpp>

using namespace proxsuite::proxqp;
using T = double;
using proxsuite::nullopt;

int
main()
{
  std::cout << "Solve a simple example with equality and inequality "
               "constraints using dense PROXQP without API"
            << std::endl;

  // Define the problem
  dense::isize dim = 3, n_eq = 3, n_in = 3;

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

  // inequality constraints C
  Eigen::MatrixXd C = Eigen::MatrixXd(n_in, dim);
  C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::VectorXd l = Eigen::VectorXd(n_in);
  l << -1.0, -1.0, -1.0;

  Eigen::VectorXd u = Eigen::VectorXd(n_in);
  u << 1.0, 1.0, 1.0;

  // Solve the problem using the dense backend
  proxsuite::proxqp::Results<T> results_dense_solver = dense::solve<T>(
    H,
    g,
    A,
    b,
    C,
    l,
    u,
    nullopt,
    nullopt,
    nullopt,
    1.e-9,
    0,
    nullopt,
    nullopt,
    nullopt,
    true,
    true,
    false,
    nullopt,
    proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
    false,
    nullopt,
    nullopt,
    false,
    nullopt);

  std::cout << "solution x: " << results_dense_solver.x << std::endl;
  std::cout << "primal residual: " << results_dense_solver.info.pri_res
            << std::endl;
  std::cout << "dual residual: " << results_dense_solver.info.dua_res
            << std::endl;
  std::cout << "duality gap: " << results_dense_solver.info.duality_gap
            << std::endl;
  std::cout << "setup timing " << results_dense_solver.info.setup_time
            << " solve time " << results_dense_solver.info.solve_time
            << std::endl;
}