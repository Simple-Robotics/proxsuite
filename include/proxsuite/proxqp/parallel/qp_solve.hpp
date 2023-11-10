//
// Copyright (c) 2023 INRIA
//
/** \file */
#ifndef PROXSUITE_PROXQP_PARALLEL_QPSOLVE_HPP
#define PROXSUITE_PROXQP_PARALLEL_QPSOLVE_HPP

// #include "proxsuite/proxqp/dense/wrapper.hpp"
#include "proxsuite/proxqp/dense/compute_ECJ.hpp"
#include "proxsuite/proxqp/sparse/wrapper.hpp"
#include "proxsuite/proxqp/parallel/omp.hpp"

namespace proxsuite {
namespace proxqp {
namespace dense {

template<typename T>
void
solve_in_parallel(std::vector<proxqp::dense::QP<T>>& qps,
                  const optional<size_t> num_threads = nullopt)
{
  size_t NUM_THREADS =
    std::max((size_t)(omp_get_max_threads() / 2),
             (size_t)(1)); // TODO(jcarpent): find optimal dispatch
  if (num_threads != nullopt) {
    NUM_THREADS = num_threads.value();
  }
  set_default_omp_options(NUM_THREADS);

  typedef proxqp::dense::QP<T> qp_dense;
  const long batch_size = long(qps.size());
  long i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_dense& qp = qps[i];
    qp.solve();
  }
}

template<typename T>
void
solve_in_parallel(proxqp::dense::BatchQP<T>& qps,
                  const optional<size_t> num_threads = nullopt)
{
  size_t NUM_THREADS =
    std::max((size_t)(omp_get_max_threads() / 2), (size_t)(1));
  if (num_threads != nullopt) {
    NUM_THREADS = num_threads.value();
  }
  set_default_omp_options(NUM_THREADS);

  typedef proxqp::dense::QP<T> qp_dense;
  const long batch_size = long(qps.size());
  long i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_dense& qp = qps[i];
    qp.solve();
  }
}

template<typename T>
void
qp_solve_in_parallel(optional<const size_t> num_threads,
                     proxqp::dense::BatchQP<T>& qps)
{
  if (num_threads != nullopt) {
    set_default_omp_options(num_threads.value());
  } else {
    size_t NUM_THREADS =
      std::max((size_t)(omp_get_max_threads() / 2 - 2), (size_t)(1));
    set_default_omp_options(NUM_THREADS);
  }
  typedef proxqp::dense::QP<T> qp_dense;
  const Eigen::DenseIndex batch_size = qps.size();
  Eigen::DenseIndex i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_dense& qp = qps[i];
    qp.solve();
  }
}

template<typename T>
void
qp_solve_backward_in_parallel(
  optional<const size_t> num_threads,
  std::vector<proxqp::dense::QP<T>>& qps,
  std::vector<proxqp::dense::Vec<T>>& loss_derivatives,
  T eps = 1.E-4,
  T rho_new = 1.E-6,
  T mu_new = 1.E-6)
{
  if (num_threads != nullopt) {
    set_default_omp_options(num_threads.value());
  } else {
    size_t NUM_THREADS =
      std::max((size_t)(omp_get_max_threads() / 2 - 2), (size_t)(1));
    set_default_omp_options(NUM_THREADS);
  }

  typedef proxqp::dense::QP<T> qp_dense;
  const Eigen::DenseIndex batch_size = qps.size();
  Eigen::DenseIndex i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_dense& qp = qps[i];
    dense::compute_backward<T>(qp, loss_derivatives[i], eps, rho_new, mu_new);
  }
}

template<typename T>
void
qp_solve_backward_in_parallel(
  optional<const size_t> num_threads,
  proxqp::dense::BatchQP<T>& qps,
  std::vector<proxqp::dense::Vec<T>>& loss_derivatives,
  T eps = 1.E-4,
  T rho_new = 1.E-6,
  T mu_new = 1.E-6)
{
  if (num_threads != nullopt) {
    set_default_omp_options(num_threads.value());
  } else {
    size_t NUM_THREADS =
      std::max((size_t)(omp_get_max_threads() / 2 - 2), (size_t)(1));
    set_default_omp_options(NUM_THREADS);
  }

  typedef proxqp::dense::QP<T> qp_dense;
  const Eigen::DenseIndex batch_size = qps.size();
  Eigen::DenseIndex i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_dense& qp = qps[i];
    dense::compute_backward<T>(qp, loss_derivatives[i], eps, rho_new, mu_new);
  }
}
} // namespace dense

namespace sparse {
template<typename T, typename I>
void
solve_in_parallel(proxqp::sparse::BatchQP<T, I>& qps,
                  const optional<size_t> num_threads = nullopt)
{
  size_t NUM_THREADS =
    std::max((size_t)(omp_get_max_threads() / 2), (size_t)(1));
  if (num_threads != nullopt) {
    NUM_THREADS = num_threads.value();
  }
  set_default_omp_options(NUM_THREADS);

  typedef proxqp::sparse::QP<T, I> qp_sparse;
  const Eigen::DenseIndex batch_size = qps.size();
  Eigen::DenseIndex i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_sparse& qp = qps[i];
    qp.solve();
  }
}
// TODO: debug
template<typename T, typename I>
void
solve_in_parallel(std::vector<proxqp::sparse::QP<T, I>>& qps,
                  const optional<size_t> num_threads = nullopt)
{
  size_t NUM_THREADS =
    std::max((size_t)(omp_get_max_threads() / 2), (size_t)(1));
  if (num_threads != nullopt) {
    NUM_THREADS = num_threads.value();
  }
  set_default_omp_options(NUM_THREADS);

  typedef proxqp::sparse::QP<T, I> qp_sparse;
  const Eigen::DenseIndex batch_size = qps.size();
  Eigen::DenseIndex i = 0;
#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < batch_size; i++) {
    qp_sparse& qp = qps[i];
    qp.solve();
  }
}
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_PARALLEL_QPSOLVE_HPP */
