#include <iostream>
#include <proxsuite/proxqp/sparse/sparse.hpp> // get the sparse API of ProxQP
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  // design a qp object using QP problem dimensions
  isize n = 10;
  isize n_eq(n / 4);
  isize n_in(n / 4);
  sparse::QP<T, isize> qp(n, n_eq, n_in);

  // assume you generate these matrices H, A and C for your QP problem

  T p = 0.15;            // level of sparsity
  T conditioning = 10.0; // conditioning level for H

  auto H = ::proxsuite::proxqp::utils::rand::sparse_positive_definite_rand(
    n, conditioning, p);
  auto A = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_in, n, p);

  // design a qp2 object using sparsity masks of H, A and C
  proxsuite::proxqp::sparse::QP<T, isize> qp2(
    H.cast<bool>(), A.cast<bool>(), C.cast<bool>());
}
