//
// Copyright (c) 2023 INRIA
//
#include <doctest.hpp>
#include <iostream>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <proxsuite/proxqp/parallel/qp_solve.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

using namespace proxsuite;
using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::utils;
using T = double;
using I = c_int;
using namespace proxsuite::linalg::sparse::tags;

DOCTEST_TEST_CASE("test parallel qp_solve for dense qps")
{
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);

  dense::isize dim = 500;
  dense::isize n_eq(10);
  dense::isize n_in(10);

  T strong_convexity_factor(1.e-2);
  int num_qps = 64;
  std::vector<proxqp::dense::QP<T>> qps;
  std::vector<proxqp::dense::QP<T>> qps_compare;

  // Generate two lists with identical QPs
  for (int i = 0; i < num_qps; i++) {
    utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    dense::QP<T> qp{ dim, n_eq, n_in };
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0.0;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);
    qps.push_back(qp);

    dense::QP<T> qp_compare{ dim, n_eq, n_in };
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0.0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp_compare.init(qp_random.H,
                    qp_random.g,
                    qp_random.A,
                    qp_random.b,
                    qp_random.C,
                    qp_random.l,
                    qp_random.u);
    qps_compare.push_back(qp_compare);
  }

  for (int i = 0; i < num_qps; i++) {
    qps[i].solve();
  }

  const size_t NUM_THREADS = (size_t)omp_get_max_threads();
  proxsuite::proxqp::dense::solve_in_parallel(qps_compare,
                                              (size_t)(NUM_THREADS / 2));

  for (int i = 0; i < num_qps; i++) {
    CHECK(qps[i].results.x == qps_compare[i].results.x);
  }
}

DOCTEST_TEST_CASE("test dense BatchQP and optional NUM_THREADS")
{
  double sparsity_factor = 0.15;
  T eps_abs = T(1e-9);

  dense::isize dim = 500;
  dense::isize n_eq(10);
  dense::isize n_in(10);

  T strong_convexity_factor(1.e-2);
  int num_qps = 64;
  std::vector<proxqp::dense::QP<T>> qps_compare;
  dense::BatchQP<T> qps_vector = dense::BatchQP<T>(num_qps);

  for (int i = 0; i < num_qps; i++) {
    auto& qp = qps_vector.init_qp_in_place(dim, n_eq, n_in);
    utils::rand::set_seed(i);
    proxqp::dense::Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0.0;
    qp.settings.verbose = false;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);

    qps_compare.emplace_back(dim, n_eq, n_in);
    auto& qp_compare = qps_compare.back();
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0.0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp_compare.init(qp_random.H,
                    qp_random.g,
                    qp_random.A,
                    qp_random.b,
                    qp_random.C,
                    qp_random.l,
                    qp_random.u);
  }

  for (int i = 0; i < num_qps; i++) {
    qps_compare[i].solve();
  }

  proxsuite::proxqp::dense::solve_in_parallel(qps_vector);

  for (int i = 0; i < num_qps; i++) {
    CHECK(qps_vector[i].results.x == qps_compare[i].results.x);
  }
}

DOCTEST_TEST_CASE("test parallel qp_solve for sparse qps")
{
  sparse::isize dim = 500;
  sparse::isize n_eq(10);
  sparse::isize n_in(10);

  T eps_abs = T(1e-9);
  T sparsity_factor = 0.15;
  T strong_convexity_factor = 0.01;

  int num_qps = 64;
  std::vector<sparse::QP<T, I>> qps;
  std::vector<sparse::QP<T, I>> qps_compare;

  // Generate two lists with identical QPs
  for (int i = 0; i < num_qps; i++) {
    utils::rand::set_seed(i);
    sparse::SparseModel<T> qp_random = utils::sparse_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

    qps.emplace_back(dim, n_eq, n_in);
    auto& qp = qps.back();
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0.0;
    qp.settings.verbose = false;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);

    qps_compare.emplace_back(dim, n_eq, n_in);
    auto& qp_compare = qps_compare.back();
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0.0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp_compare.init(qp_random.H,
                    qp_random.g,
                    qp_random.A,
                    qp_random.b,
                    qp_random.C,
                    qp_random.l,
                    qp_random.u);
  }

  for (int i = 0; i < num_qps; i++) {
    qps[i].solve();
  }

  const size_t NUM_THREADS = (size_t)omp_get_max_threads();
  proxsuite::proxqp::sparse::solve_in_parallel(qps_compare,
                                               (size_t)(NUM_THREADS / 2));

  for (int i = 0; i < num_qps; i++) {
    CHECK(qps[i].results.x == qps_compare[i].results.x);
  }
}

DOCTEST_TEST_CASE("test sparse BatchQP")
{
  sparse::isize dim = 500;
  sparse::isize n_eq(10);
  sparse::isize n_in(10);

  T eps_abs = T(1e-9);
  T sparsity_factor = 0.15;
  T strong_convexity_factor = 0.01;

  int num_qps = 64;
  std::vector<sparse::QP<T, I>> qps_compare;

  sparse::BatchQP<T, I> qps_vector = sparse::BatchQP<T, I>(num_qps);
  // qps_vector.init_qp_in_place(dim, n_eq, n_in);
  // Generate two lists with identical QPs
  for (int i = 0; i < num_qps; i++) {
    auto& qp = qps_vector.init_qp_in_place(dim, n_eq, n_in);
    utils::rand::set_seed(i);
    sparse::SparseModel<T> qp_random = utils::sparse_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp.settings.eps_abs = eps_abs;
    qp.settings.eps_rel = 0.0;
    qp.settings.verbose = false;
    qp.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp.init(qp_random.H,
            qp_random.g,
            qp_random.A,
            qp_random.b,
            qp_random.C,
            qp_random.l,
            qp_random.u);

    qps_compare.emplace_back(dim, n_eq, n_in);
    auto& qp_compare = qps_compare.back();
    qp_compare.settings.eps_abs = eps_abs;
    qp_compare.settings.eps_rel = 0.0;
    qp_compare.settings.initial_guess = InitialGuessStatus::NO_INITIAL_GUESS;
    qp_compare.init(qp_random.H,
                    qp_random.g,
                    qp_random.A,
                    qp_random.b,
                    qp_random.C,
                    qp_random.l,
                    qp_random.u);
  }

  for (int i = 0; i < num_qps; i++) {
    qps_compare[i].solve();
  }

  const size_t NUM_THREADS = (size_t)omp_get_max_threads();
  proxsuite::proxqp::sparse::solve_in_parallel(qps_vector,
                                               (size_t)(NUM_THREADS / 2));

  for (int i = 0; i < num_qps; i++) {
    CHECK(qps_vector[i].results.x == qps_compare[i].results.x);
  }
}
