#include "qp/sparse/sparse.hpp"  // get the sparse API of ProxQP
#include "test/include/util.hpp" // use a function for generating a random QP

using namespace qp;
using T = double;
using I = c_int;
int
main()
{
  // design a Qp object using QP problem dimensions
  I n = 10;
  I n_eq(n / 4);
  I n_in(n / 4);
  sparse::QP<T, I> Qp(n, n_eq, n_in);

  // assume you generate these matrices H, A and C for your QP problem

  T p = 0.15;            // level of sparsity
  T conditioning = 10.0; // conditioning level for H

  auto H = ldlt_test::rand::sparse_positive_definite_rand(n, conditioning, p);
  auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in, n, p);

  // design a Qp2 object using sparsity masks of H, A and C
  qp::sparse::QP<T, I> Qp2(H.cast<bool>(), A.cast<bool>(), C.cast<bool>());
}
