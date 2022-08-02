#include "proxsuite/proxqp/sparse/sparse.hpp" // get the sparse backend of ProxQP
#include "proxsuite/proxqp/dense/dense.hpp"   // get the dense backend of ProxQP
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>// used for generating a random convex Qp

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  isize n = 10;
  isize n_eq(n / 4);
  isize n_in(n / 4);
  T p = 0.15;            // level of sparsity
  T conditioning = 10.0; // conditioning level for H
  auto H = utils::rand::sparse_positive_definite_rand(n, conditioning, p);
  auto g = utils::rand::vector_rand<T>(n);
  auto A = utils::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = utils::rand::sparse_matrix_rand<T>(n_in, n, p);
  auto x_sol = utils::rand::vector_rand<T>(n);
  auto b = (A * x_sol).eval();
  auto l = (C * x_sol).eval();
  auto u = (l.array() + 10).matrix().eval();
  // Solve the problem using the sparse backend
  Results<T> results_sparse_solver =
    sparse::solve<T, isize>(H, g, A, b, C, u, l);
  // Solve the problem using the dense backend
  Results<T> results_dense_solver = dense::solve<T>(H, g, A, b, C, u, l);
}
