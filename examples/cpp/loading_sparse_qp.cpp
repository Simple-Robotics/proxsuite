#include "proxsuite/proxqp/sparse/sparse.hpp" // get the sparse API of ProxQP
#include "util.hpp" // use a function for generating a random QP

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  // design a Qp object using QP problem dimensions
  isize n = 10;
  isize n_eq(n / 4);
  isize n_in(n / 4);
  sparse::QP<T, isize> Qp(n, n_eq, n_in);

  // assume you generate these matrices H, A and C for your QP problem

  T p = 0.15;            // level of sparsity
  T conditioning = 10.0; // conditioning level for H

  auto H = ::proxsuite::proxqp::test::rand::sparse_positive_definite_rand(
    n, conditioning, p);
  auto A = ::proxsuite::proxqp::test::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = ::proxsuite::proxqp::test::rand::sparse_matrix_rand<T>(n_in, n, p);

  // design a Qp2 object using sparsity masks of H, A and C
  proxsuite::proxqp::sparse::QP<T, isize> Qp2(
    H.cast<bool>(), A.cast<bool>(), C.cast<bool>());
}
