#include "proxsuite/proxqp/sparse/sparse.hpp" // get the sparse backend of ProxQP
#include "proxsuite/proxqp/dense/dense.hpp"   // get the dense backend of ProxQP
#include "util.hpp" // use a function for generating a random QP

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
  auto H = test::rand::sparse_positive_definite_rand(n, conditioning, p);
  auto g = test::rand::vector_rand<T>(n);
  auto A = test::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = test::rand::sparse_matrix_rand<T>(n_in, n, p);
  auto x_sol = test::rand::vector_rand<T>(n);
  auto b = (A * x_sol).eval();
  auto l = (C * x_sol).eval();
  auto u = (l.array() + 10).matrix().eval();
  // Solve the problem using the sparse backend
  Results<T> results_sparse_solver =
    sparse::solve<T, isize>(H, g, A, b, C, u, l);
  // Solve the problem using the dense backend
  Results<T> results_dense_solver = dense::solve<T>(H, g, A, b, C, u, l);
}
