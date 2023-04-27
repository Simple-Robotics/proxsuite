//
// Copyright (c) 2023 INRIA
//
#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/parallel/qp_solve.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using T = double;
using namespace proxsuite;
using namespace proxsuite::proxqp;

int
main(int /*argc*/, const char** /*argv*/)
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);

  // dense::isize dim = 500;
  // dense::isize n_eq(200);
  // dense::isize n_in(200);

  dense::isize dim = 100;
  dense::isize n_eq(50);
  dense::isize n_in(50);

  T strong_convexity_factor(1.e-2);
  int num_qps = 1024;
  const int smooth = 10;
  std::vector<proxqp::dense::QP<T>> qps;

  std::cout << "--" << std::endl;
  std::cout << "dim: " << dim << std::endl;
  std::cout << "n_eq: " << n_eq << std::endl;
  std::cout << "n_in: " << n_in << std::endl;
  std::cout << "--" << std::endl;
  std::cout << "batch_size: " << num_qps << std::endl;
  std::cout << "--" << std::endl;

  // Generate and initialize Qps
  for (int i = 0; i < num_qps; i++) {
    utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    dense::QP<T> qp{ dim, n_eq, n_in };
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qps.push_back(qp);
  }
  std::cout << "Generation done.\n" << std::endl;

  Timer<T> timer;
  for (int j = 0; j < smooth; j++) {
    for (int i = 0; i < num_qps; i++) {
      qps[i].solve();
    }
  }
  timer.stop();
  std::cout << "mean solve time in serial: \t\t\t"
            << timer.elapsed().user * 1e-3 / smooth << "ms" << std::endl;

  const size_t NUM_THREADS = (size_t)omp_get_max_threads();
  for (size_t num_threads = 1; num_threads <= NUM_THREADS; ++num_threads) {
    Timer<T> timer;
    for (int j = 0; j < smooth; j++) {
      proxsuite::proxqp::parallel::qp_solve_in_parallel(num_threads, qps);
    }
    timer.stop();
    std::cout << "mean solve time in parallel (" << num_threads
              << " threads):\t" << timer.elapsed().user * 1e-3 / smooth << "ms"
              << std::endl;
  }
}
