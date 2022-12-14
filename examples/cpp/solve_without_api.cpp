#include <iostream>
#include "proxsuite/proxqp/sparse/sparse.hpp" // get the sparse backend of ProxQP
#include "proxsuite/proxqp/dense/dense.hpp"   // get the dense backend of ProxQP
#include "proxsuite/proxqp/utils/random_qp_problems.hpp" // used for generating a random convex qp

using namespace proxsuite::proxqp;
using T = double;
using Mat = dense::Mat<T>;
using Vec = dense::Vec<T>;

int
main()
{
  isize n = 10;
  isize n_eq(n / 4);
  isize n_in(n / 4);
  T p = 0.35;            // level of sparsity
  T conditioning = 10.0; // conditioning level for H
  auto H = utils::rand::sparse_positive_definite_rand(
    n, conditioning, p); // upper triangular matrix
  Mat H_dense = Mat(H);
  H_dense.template triangularView<Eigen::Lower>() = H_dense.transpose();
  Vec g = utils::rand::vector_rand<T>(n);
  auto A = utils::rand::sparse_matrix_rand<T>(n_eq, n, p);
  Mat A_dense = Mat(A);
  auto C = utils::rand::sparse_matrix_rand<T>(n_in, n, p);
  Mat C_dense = Mat(C);
  Vec x_sol = utils::rand::vector_rand<T>(n);
  Vec b = A * x_sol;
  Vec l = C * x_sol;
  Vec u = (l.array() + 10).matrix();

  // Solve the problem using the sparse backend
  Results<T> results_sparse_solver =
    sparse::solve<T, isize>(H, g, A, b, C, l, u);
  std::cout << "optimal x from sparse solver: " << results_sparse_solver.x
            << std::endl;
  std::cout << "optimal y from sparse solver: " << results_sparse_solver.y
            << std::endl;
  std::cout << "optimal z from sparse solver: " << results_sparse_solver.z
            << std::endl;
  // Solve the problem using the dense backend
  Results<T> results_dense_solver =
    dense::solve<T>(H_dense, g, A_dense, b, C_dense, l, u);

  // print an optimal solution x,y and z
  std::cout << "optimal x from dense solver: " << results_dense_solver.x
            << std::endl;
  std::cout << "optimal y from dense solver: " << results_dense_solver.y
            << std::endl;
  std::cout << "optimal z from dense solver: " << results_dense_solver.z
            << std::endl;
}
