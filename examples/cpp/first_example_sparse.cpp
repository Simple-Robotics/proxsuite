//
// Copyright (c) 2022 INRIA
//
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <proxsuite/helpers/optional.hpp> // for c++14
#include <proxsuite/proxqp/sparse/sparse.hpp>

using namespace proxsuite::proxqp;
using proxsuite::nullopt; // c++17 simply use std::nullopt

int
main()
{

  std::cout
    << "Solve a simple example with inequality constraints using sparse ProxQP"
    << std::endl;

  // define the problem
  double eps_abs = 1e-9;
  sparse::isize dim = 3, n_eq = 0, n_in = 3, nnz = 2;

  // random utils to generate sparse matrix
  std::random_device rd;  // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> int_distr(
    0, static_cast<int>(dim) - 1); // define the range
  std::uniform_real_distribution<> double_distr(0, 1);

  // cost H: first way to define sparse matrix from triplet list
  std::vector<Eigen::Triplet<double>>
    coefficients; // list of non-zeros coefficients
  coefficients.reserve(nnz);

  for (sparse::isize k = 0; k < nnz; k++) {
    int col = int_distr(gen);
    int row = int_distr(gen);
    double val = double_distr(gen);
    coefficients.push_back(Eigen::Triplet<double>(col, row, val));
  }
  Eigen::SparseMatrix<double> H_spa(dim, dim);
  H_spa.setFromTriplets(coefficients.begin(), coefficients.end());

  Eigen::VectorXd g = Eigen::VectorXd::Random(dim);

  // inequality constraints C: other way to define sparse matrix from  dense
  // matrix using sparseView
  Eigen::MatrixXd C = Eigen::MatrixXd(n_in, dim);
  C << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::SparseMatrix<double> C_spa(n_in, dim);
  C_spa = C.sparseView();

  Eigen::VectorXd l = Eigen::VectorXd(n_in);
  l << -1.0, -1.0, -1.0;

  Eigen::VectorXd u = Eigen::VectorXd(n_in);
  u << 1.0, 1.0, 1.0;

  std::cout << "H:\n" << H_spa << std::endl;
  std::cout << "g.T:" << g.transpose() << std::endl;
  std::cout << "C:\n" << C_spa << std::endl;
  std::cout << "l.T:" << l.transpose() << std::endl;
  std::cout << "u.T:" << u.transpose() << std::endl;

  // create qp object and pass some settings
  sparse::QP<double, long long> qp(dim, n_eq, n_in);

  qp.settings.eps_abs = eps_abs;
  qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
  qp.settings.verbose = true;

  // initialize qp with matrices describing the problem
  // note: it is also possible to use update here
  qp.init(H_spa, g, nullopt, nullopt, C_spa, l, u);

  qp.solve();

  std::cout << "primal residual: " << qp.results.info.pri_res << std::endl;
  std::cout << "dual residual: " << qp.results.info.dua_res << std::endl;
  std::cout << "total number of iteration: " << qp.results.info.iter
            << std::endl;
  std::cout << "setup timing " << qp.results.info.setup_time << " solve time "
            << qp.results.info.solve_time << std::endl;
  return 0;
}