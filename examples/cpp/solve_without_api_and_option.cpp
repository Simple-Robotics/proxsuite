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
  auto H = ::proxsuite::proxqp::test::rand::sparse_positive_definite_rand(
    n, conditioning, p);
  auto g = ::proxsuite::proxqp::test::rand::vector_rand<T>(n);
  auto A = ::proxsuite::proxqp::test::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = ::proxsuite::proxqp::test::rand::sparse_matrix_rand<T>(n_in, n, p);
  auto x_sol = ::proxsuite::proxqp::test::rand::vector_rand<T>(n);
  auto b = A * x_sol;
  auto l = C * x_sol;
  auto u = (l.array() + 10).matrix().eval();
  // Solve the problem using the dense backend
  // and suppose you want to change the accuracy to 1.E-9 and rho initial value
  // to 1.E-7
  proxsuite::proxqp::Results<T> results =
    proxsuite::proxqp::dense::solve<T>(H,
                                       g,
                                       A,
                                       b,
                                       C,
                                       u,
                                       l,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       T(1.E-9),
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       T(1.E-7));
}
