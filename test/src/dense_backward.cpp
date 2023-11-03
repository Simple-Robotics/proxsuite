//
// Copyright (c) 2023 INRIA
//
#include <iostream>
#include <doctest.hpp>
#include <Eigen/Core>
#include <optional>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/proxqp/dense/compute_ECJ.hpp>

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

DOCTEST_TEST_CASE("proxqp::dense: test compute backward for g (feasible QP)")
{
  double sparsity_factor = 0.85;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(5), n_in(0);
  T strong_convexity_factor(1.e-1);
  proxqp::dense::Model<T> random_qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  Eigen::Matrix<T, 10, 10> H = random_qp.H;
  Eigen::Matrix<T, 10, 1> g = random_qp.g;
  Eigen::Matrix<T, 5, 10> A = random_qp.A;
  Eigen::Matrix<T, 5, 1> b = random_qp.b;
  // Eigen::Matrix<T, 2, 10> C = random_qp.C;
  // Eigen::Matrix<T, 2, 1> l = random_qp.l;
  // Eigen::Matrix<T, 2, 1> u = random_qp.u;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;

  qp.init(H, g, A, b, nullopt, nullopt, nullopt);
  qp.solve();

  // Compute dx_dg using backward function
  Eigen::VectorXd loss_derivative = Eigen::VectorXd::Zero(dim + n_eq + n_in);
  Eigen::MatrixXd dx_dg = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; i++) {
    loss_derivative(i) = T(1);
    dense::compute_backward<double>(qp, loss_derivative, 1e-5, 1e-7, 1e-7);
    dx_dg.row(i) = qp.model.backward_data.dL_dg;
    loss_derivative(i) = T(0);
  }
  std::cout << "dx_dg: " << std::endl << dx_dg << std::endl;

  // Compute dx_dg using finite differences
  Eigen::MatrixXd dx_dg_fd = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::VectorXd g_fd = g;
  T eps = 1e-5;
  for (int i = 0; i < g.size(); i++) {
    g_fd(i) += eps;
    qp.init(H, g_fd, A, b, nullopt, nullopt, nullopt);
    qp.solve();
    Eigen::VectorXd x_plus = qp.results.x;
    g_fd(i) = g(i);
    g_fd(i) -= eps;
    qp.init(H, g_fd, A, b, nullopt, nullopt, nullopt);
    qp.solve();
    Eigen::VectorXd x_minus = qp.results.x;

    g_fd(i) = T(0);
    dx_dg_fd.col(i) = (x_plus - x_minus) / (T(2) * eps);
  }
  std::cout << "dx_dg_fd: " << std::endl << dx_dg_fd << std::endl;

  // Compare dx_dg_fd with the result from the backward function
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      DOCTEST_CHECK(std::abs(dx_dg_fd(i, j) - dx_dg(i, j)) < 1e-5);
    }
  }
}

DOCTEST_TEST_CASE("proxqp::dense: test compute backward for b (feasible QP)")
{
  double sparsity_factor = 0.85;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 10;

  dense::isize n_eq(5), n_in(0);
  T strong_convexity_factor(1.e-2);
  proxqp::dense::Model<T> random_qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  Eigen::Matrix<T, 10, 10> H = random_qp.H;
  Eigen::Matrix<T, 10, 1> g = random_qp.g;
  Eigen::Matrix<T, 5, 10> A = random_qp.A;
  Eigen::Matrix<T, 5, 1> b = random_qp.b;
  // Eigen::Matrix<T, 2, 10> C = random_qp.C;
  // Eigen::Matrix<T, 2, 1> l = random_qp.l;
  // Eigen::Matrix<T, 2, 1> u = random_qp.u;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;

  qp.init(H, g, A, b, nullopt, nullopt, nullopt);
  qp.solve();

  // Compute dx_db using backward function
  Eigen::VectorXd loss_derivative = Eigen::VectorXd::Zero(dim + n_eq + n_in);
  Eigen::MatrixXd dx_db = Eigen::MatrixXd::Zero(dim, n_eq);
  for (int i = 0; i < dim; i++) {
    loss_derivative(i) = 1;
    dense::compute_backward<double>(qp, loss_derivative, 1e-5, 1e-7, 1e-7);
    dx_db.row(i) = qp.model.backward_data.dL_db;
    loss_derivative(i) = 0;
  }
  std::cout << "dx_db: " << std::endl << dx_db << std::endl;

  // Compute dx_db using finite differences
  Eigen::MatrixXd dx_db_fd = Eigen::MatrixXd::Zero(dim, n_eq);
  Eigen::VectorXd b_fd = b;
  T eps = 1e-5;
  for (int i = 0; i < b.size(); i++) {
    b_fd(i) += eps;
    qp.init(H, g, A, b_fd, nullopt, nullopt, nullopt);
    qp.solve();
    Eigen::VectorXd x_plus = qp.results.x;

    b_fd(i) -= 2 * eps;
    qp.init(H, g, A, b_fd, nullopt, nullopt, nullopt);
    qp.solve();
    Eigen::VectorXd x_minus = qp.results.x;

    b_fd(i) += eps;
    dx_db_fd.col(i) = (x_plus - x_minus) / (2 * eps);
  }
  std::cout << "dx_db_fd: " << std::endl << dx_db_fd << std::endl;

  // Compare dx_db_fd with the result from the backward function
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < n_eq; j++) {
      DOCTEST_CHECK(std::abs(dx_db_fd(i, j) - dx_db(i, j)) < 1e-5);
    }
  }
}

DOCTEST_TEST_CASE("proxqp::dense: test compute backward for g (QP with "
                  "saturating inequality constraints)")
{
  double sparsity_factor = 0.85;
  T eps_abs = T(1e-9);
  utils::rand::set_seed(1);
  dense::isize dim = 6;

  dense::isize n_eq(0), n_in(12);
  T strong_convexity_factor(1.e-1);
  proxqp::dense::Model<T> random_qp = proxqp::utils::dense_strongly_convex_qp(
    dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  std::cout << "creating random  qp " << std::endl;
  Eigen::Matrix<T, 6, 6> H = random_qp.H;
  Eigen::Matrix<T, 6, 1> g = random_qp.g;
  // Eigen::Matrix<T, 5, 10> A = random_qp.A;
  // Eigen::Matrix<T, 5, 1> b = random_qp.b;
  Eigen::Matrix<T, 12, 6> C = random_qp.C;
  Eigen::Matrix<T, 12, 1> l = random_qp.l;
  l(0) = 1e3;
  l(1) = 1e3;
  l(2) = 1e3;
  l(3) = 1e3;
  l(9) = 1e3;
  // Eigen::Matrix<T, 2, 1> u = random_qp.u;

  dense::QP<T> qp{ dim, n_eq, n_in }; // creating QP object
  qp.settings.eps_abs = eps_abs;
  qp.settings.eps_rel = 0;

  qp.init(H, g, nullopt, nullopt, C, l, nullopt);
  std::cout << "solving qp " << std::endl;
  qp.solve();
  std::cout << "active ineq  " << qp.work.active_inequalities.count()
            << std::endl;

  // Compute dx_dg using backward function
  Eigen::VectorXd loss_derivative = Eigen::VectorXd::Zero(dim + n_eq + n_in);
  Eigen::MatrixXd dx_dg = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; i++) {
    loss_derivative(i) = T(1);
    dense::compute_backward<double>(qp, loss_derivative, 1e-5, 1e-7, 1e-7);
    dx_dg.row(i) = qp.model.backward_data.dL_dg;
    loss_derivative(i) = T(0);
  }
  std::cout << "dx_dg: " << std::endl << dx_dg << std::endl;

  // Compute dx_dg using finite differences
  Eigen::MatrixXd dx_dg_fd = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::VectorXd g_fd = g;
  T eps = 1e-5;
  for (int i = 0; i < g.size(); i++) {
    g_fd(i) += eps;
    qp.init(H, g_fd, nullopt, nullopt, C, l, nullopt);
    qp.solve();
    Eigen::VectorXd x_plus = qp.results.x;
    g_fd(i) = g(i);
    g_fd(i) -= eps;
    qp.init(H, g_fd, nullopt, nullopt, C, l, nullopt);
    qp.solve();
    Eigen::VectorXd x_minus = qp.results.x;

    g_fd(i) = T(0);
    dx_dg_fd.col(i) = (x_plus - x_minus) / (T(2) * eps);
  }
  std::cout << "dx_dg_fd: " << std::endl << dx_dg_fd << std::endl;

  // Compare dx_dg_fd with the result from the backward function
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      DOCTEST_CHECK(std::abs(dx_dg_fd(i, j) - dx_dg(i, j)) < 1e-5);
    }
  }
}