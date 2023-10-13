/*

Compile this code once with and without vectorization to see performance
difference.

Use vectorization:
g++ -O3 -march=native -DNDEBUG -std=gnu++17 -DPROXSUITE_VECTORIZE
benchmark_dense_qp.cpp -o benchmark_dense_qp $(pkg-config --cflags proxsuite)

Do not use vectorization:
g++ -DNDEBUG -std=gnu++17 benchmark_dense_qp.cpp -o benchmark_dense_qp
$(pkg-config --cflags proxsuite)

Comparison of timings on Intel i7-11850H and ubuntu 20.04 using this file:

With vectorization:
sparsity_factor: 0.1
Setup Time consumption(dense): 0.000237295s
Solve Time consumption(dense): 0.000500206s
sparsity_factor: 0.2
Setup Time consumption(dense): 0.000465961s
Solve Time consumption(dense): 0.000903787s
sparsity_factor: 0.3
Setup Time consumption(dense): 0.000697931s
Solve Time consumption(dense): 0.00136976s
sparsity_factor: 0.4
Setup Time consumption(dense): 0.000931736s
Solve Time consumption(dense): 0.00185252s


Without vectorization:
sparsity_factor: 0.1
Setup Time consumption(dense): 0.0147825s
Solve Time consumption(dense): 0.0277815s
sparsity_factor: 0.2
Setup Time consumption(dense): 0.029592s
Solve Time consumption(dense): 0.0490869s
sparsity_factor: 0.3
Setup Time consumption(dense): 0.0443664s
Solve Time consumption(dense): 0.0746045s
sparsity_factor: 0.4
Setup Time consumption(dense): 0.0592621s
Solve Time consumption(dense): 0.101507s

*/
#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using namespace proxsuite::proxqp;
using T = double;

int
main()
{
  double N = 100;
  double solve_time = 0.0;
  double setup_time = 0.0;

  dense::isize dim = 100;
  dense::isize n_eq(dim / 2);
  dense::isize n_in(dim / 2);

  for (T sparsity_factor = 0.1; sparsity_factor < 0.5; sparsity_factor += 0.1) {
    T strong_convexity_factor(1.e-2);
    dense::Model<T> qp_random = utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    for (int i = 0; i < N; i++) {
      dense::QP<T> qp(dim, n_eq, n_in);
      qp.settings.compute_timings = true; // compute all timings
      qp.settings.max_iter = 10000;
      qp.settings.max_iter_in = 1000;
      qp.settings.eps_abs = 1e-5;
      qp.settings.eps_rel = 0;
      qp.init(qp_random.H,
              qp_random.g,
              qp_random.A,
              qp_random.b,
              qp_random.C,
              qp_random.l,
              qp_random.u);
      qp.solve();
      solve_time += qp.results.info.solve_time / N;
      setup_time += qp.results.info.setup_time / N;
    }
    std::cout << "sparsity_factor: " << sparsity_factor << std::endl;
    std::cout << "Setup Time consumption(dense): " << setup_time / 1e6 << "s"
              << std::endl
              << "Solve Time consumption(dense): " << solve_time / 1e6 << "s"
              << std::endl;
  }

  return 0;
}