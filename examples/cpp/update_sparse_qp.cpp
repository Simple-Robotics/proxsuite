#include <iostream>
#include <proxsuite/proxqp/sparse/sparse.hpp> // get the sparse API of ProxQP
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp

using namespace proxsuite;
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
  auto H = ::proxsuite::proxqp::utils::rand::sparse_positive_definite_rand(
    n, conditioning, p);
  auto g = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
  auto A = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_eq, n, p);
  auto C = ::proxsuite::proxqp::utils::rand::sparse_matrix_rand<T>(n_in, n, p);
  auto x_sol = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
  auto b = A * x_sol;
  auto l = C * x_sol;
  auto u = (l.array() + 10).matrix().eval();
  // design a qp2 object using sparsity masks of H, A and C
  proxsuite::proxqp::sparse::QP<T, int> qp(
    H.cast<bool>(), A.cast<bool>(), C.cast<bool>());
  qp.init(H, g, A, b, C, l, u);
  qp.solve();
  // update H
  auto H_new = 2 * H; // keep the same sparsity structure
  qp.update(H_new,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt); // update H with H_new, it will work
  qp.solve();
  // generate H2 with another sparsity structure
  auto H2 = ::proxsuite::proxqp::utils::rand::sparse_positive_definite_rand(
    n, conditioning, p);
  qp.update(H2,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt); // nothing will happen
  // if only a vector changes, then the update takes effect
  auto g_new = ::proxsuite::proxqp::utils::rand::vector_rand<T>(n);
  qp.update(nullopt, g, nullopt, nullopt, nullopt, nullopt, nullopt);
  qp.solve(); // it solves the problem with another vector
  // to solve the problem with H2 matrix create a new qp object
  proxsuite::proxqp::sparse::QP<T, isize> qp2(
    H2.cast<bool>(), A.cast<bool>(), C.cast<bool>());
  qp2.init(H2, g_new, A, b, C, l, u);
  qp2.solve(); // it will solve the new problem
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp2.results.x << std::endl;
  std::cout << "optimal y: " << qp2.results.y << std::endl;
  std::cout << "optimal z: " << qp2.results.z << std::endl;
}
