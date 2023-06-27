//
// Copyright (c) 2023 INRIA
//
#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/parallel/qp_solve.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using T = double;
using I = long long;

using namespace proxsuite;
using namespace proxsuite::proxqp;

int
main(int /*argc*/, const char** /*argv*/)
{

  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);

  dense::isize dim = 100;
  dense::isize n_eq(50);
  dense::isize n_in(50);

  T strong_convexity_factor(1.e-2);
  int num_qps = 1024;
  int smooth = 10;

  std::cout << "#################### Problem dimensions #################### "
            << std::endl;
  std::cout << "dim: " << dim << std::endl;
  std::cout << "n_eq: " << n_eq << std::endl;
  std::cout << "n_in: " << n_in << std::endl;
  std::cout << "batch_size: " << num_qps << std::endl;

  {
    // Benchmark: generate and initialize dense qps
    std::cout << "############### Generating and initializing QPs ############"
              << std::endl;
    Timer<T> timer;
    for (int j = 0; j < smooth; j++) {
      std::vector<proxqp::dense::QP<T>> qps;
      qps.reserve(num_qps);
      for (int i = 0; i < num_qps; i++) {
        utils::rand::set_seed(i);
        proxqp::dense::Model<T> qp_random =
          proxqp::utils::dense_strongly_convex_qp(
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
        qps.push_back(std::move(qp));
      }
    }
    timer.stop();
    std::cout << "time to generate and initialize std::vector of dense qps: \t"
              << timer.elapsed().user * 1e-3 / smooth << " ms" << std::endl;

    timer.start();
    for (int j = 0; j < smooth; j++) {
      dense::BatchQP<T> qps_vector = dense::BatchQP<T>(num_qps);
      for (int i = 0; i < num_qps; i++) {
        utils::rand::set_seed(i);
        proxqp::dense::Model<T> qp_random =
          proxqp::utils::dense_strongly_convex_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

        auto& qp = qps_vector.init_qp_in_place(dim, n_eq, n_in);
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
      }
    }
    timer.stop();
    std::cout << "time to generate and initialize dense:BatchQP: \t\t\t"
              << timer.elapsed().user * 1e-3 / smooth << " ms" << std::endl;

    for (int j = 0; j < smooth; j++) {
      std::vector<proxqp::sparse::QP<T, I>> qps;
      qps.reserve(num_qps);
      for (int i = 0; i < num_qps; i++) {
        utils::rand::set_seed(i);
        proxqp::dense::Model<T> qp_dense =
          proxqp::utils::dense_strongly_convex_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
        proxqp::sparse::SparseModel<T> qp_random = qp_dense.to_sparse();

        sparse::QP<T, I> qp{ dim, n_eq, n_in };
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
        qps.push_back(std::move(qp));
      }
    }
    timer.stop();
    std::cout << "time to generate and initialize std::vector of sparse qps: \t"
              << timer.elapsed().user * 1e-3 / smooth << " ms" << std::endl;

    timer.start();
    for (int j = 0; j < smooth; j++) {
      sparse::BatchQP<T, I> qps_vector = sparse::BatchQP<T, I>(num_qps);
      for (int i = 0; i < num_qps; i++) {
        utils::rand::set_seed(i);
        proxqp::dense::Model<T> qp_dense =
          proxqp::utils::dense_strongly_convex_qp(
            dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
        proxqp::sparse::SparseModel<T> qp_random = qp_dense.to_sparse();

        auto& qp = qps_vector.init_qp_in_place(dim, n_eq, n_in);
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
      }
    }
    timer.stop();
    std::cout << "time to generate and initialize sparse:BatchQP: \t\t\t"
              << timer.elapsed().user * 1e-3 / smooth << " ms" << std::endl;
  }

  {
    // Benchmark: solve dense qps
    Timer<T> timer;
    std::cout << "#################### Solving DENSE QPs #################### "
              << std::endl;
    std::vector<proxqp::dense::QP<T>> qps;
    qps.reserve(num_qps);
    for (int i = 0; i < num_qps; i++) {
      utils::rand::set_seed(i);
      proxqp::dense::Model<T> qp_random =
        proxqp::utils::dense_strongly_convex_qp(
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
      qps.push_back(std::move(qp));
    }

    dense::BatchQP<T> qps_vector = dense::BatchQP<T>(num_qps);
    for (int i = 0; i < num_qps; i++) {
      utils::rand::set_seed(i);
      proxqp::dense::Model<T> qp_random =
        proxqp::utils::dense_strongly_convex_qp(
          dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

      auto& qp = qps_vector.init_qp_in_place(dim, n_eq, n_in);
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
    }

    timer.start();
    for (int j = 0; j < smooth; j++) {
      for (int i = 0; i < num_qps; i++) {
        qps[i].solve();
      }
    }
    timer.stop();
    std::cout << "mean solve time in serial: \t\t\t"
              << timer.elapsed().user * 1e-3 / smooth << " ms" << std::endl;

    const size_t NUM_THREADS = (size_t)omp_get_max_threads();

    std::cout << "\nparallel using dense::BatchQP" << std::endl;
    for (size_t num_threads = 1; num_threads <= NUM_THREADS; ++num_threads) {
      timer.start();
      for (int j = 0; j < smooth; j++) {
        proxsuite::proxqp::dense::solve_in_parallel(qps_vector, num_threads);
      }
      timer.stop();
      std::cout << "mean solve time in parallel (" << num_threads
                << " threads):\t" << timer.elapsed().user * 1e-3 / smooth
                << " ms" << std::endl;
    }

    std::cout << "\nparallel using std::vector of dense QPs" << std::endl;
    for (size_t num_threads = 1; num_threads <= NUM_THREADS; ++num_threads) {
      timer.start();
      for (int j = 0; j < smooth; j++) {
        proxsuite::proxqp::dense::solve_in_parallel(qps, num_threads);
      }
      timer.stop();
      std::cout << "mean solve time in parallel (" << num_threads
                << " threads):\t" << timer.elapsed().user * 1e-3 / smooth
                << " ms" << std::endl;
    }
  }

  {
    // Benchmark: solve sparse qps
    Timer<T> timer;
    std::cout << "#################### Solving SPARSE QPs #################### "
              << std::endl;
    std::vector<proxqp::sparse::QP<T, I>> qps;
    qps.reserve(num_qps);
    for (int i = 0; i < num_qps; i++) {
      utils::rand::set_seed(i);
      proxqp::dense::Model<T> qp_dense =
        proxqp::utils::dense_strongly_convex_qp(
          dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
      proxqp::sparse::SparseModel<T> qp_random = qp_dense.to_sparse();

      sparse::QP<T, I> qp{ dim, n_eq, n_in };
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
      qps.push_back(std::move(qp));
    }

    sparse::BatchQP<T, I> qps_vector = sparse::BatchQP<T, I>(num_qps);
    for (int i = 0; i < num_qps; i++) {
      utils::rand::set_seed(i);
      proxqp::dense::Model<T> qp_dense =
        proxqp::utils::dense_strongly_convex_qp(
          dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
      proxqp::sparse::SparseModel<T> qp_random = qp_dense.to_sparse();

      auto& qp = qps_vector.init_qp_in_place(dim, n_eq, n_in);
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
    }

    timer.start();
    for (int j = 0; j < smooth; j++) {
      for (int i = 0; i < num_qps; i++) {
        qps[i].solve();
      }
    }
    timer.stop();
    std::cout << "mean solve time in serial: \t\t\t"
              << timer.elapsed().user * 1e-3 / smooth << " ms" << std::endl;

    const size_t NUM_THREADS = (size_t)omp_get_max_threads();

    std::cout << "\nparallel using sparse::BatchQP" << std::endl;
    for (size_t num_threads = 1; num_threads <= NUM_THREADS; ++num_threads) {
      timer.start();
      for (int j = 0; j < smooth; j++) {
        proxsuite::proxqp::sparse::solve_in_parallel(qps_vector, num_threads);
      }
      timer.stop();
      std::cout << "mean solve time in parallel (" << num_threads
                << " threads):\t" << timer.elapsed().user * 1e-3 / smooth
                << " ms" << std::endl;
    }

    std::cout << "\nparallel using std::vector of sparse QPs" << std::endl;
    for (size_t num_threads = 1; num_threads <= NUM_THREADS; ++num_threads) {
      timer.start();
      for (int j = 0; j < smooth; j++) {
        proxsuite::proxqp::sparse::solve_in_parallel(qps, num_threads);
      }
      timer.stop();
      std::cout << "mean solve time in parallel (" << num_threads
                << " threads):\t" << timer.elapsed().user * 1e-3 / smooth
                << " ms" << std::endl;
    }
  }
}
