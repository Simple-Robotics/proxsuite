//
// Copyright (c) 2022 INRIA
//
/** \file */
#ifndef PROXSUITE_PROXQP_SPARSE_HELPERS_HPP
#define PROXSUITE_PROXQP_SPARSE_HELPERS_HPP

#include <Eigen/Sparse>
#include <proxsuite/helpers/optional.hpp>

#include <proxsuite/linalg/veg/vec.hpp>
#include <proxsuite/proxqp/sparse/fwd.hpp>
#include <iostream>
namespace proxsuite {
namespace proxqp {
namespace sparse {

template<typename T, typename I>
T
power_iteration(SparseMat<T, I>& H,
                sparse::Vec<T>& dw,
                sparse::Vec<T>& rhs,
                sparse::Vec<T>& err_v,
                T power_iteration_accuracy,
                isize nb_power_iteration)
{
  // computes maximal eigen value of the bottom right matrix of the LDLT
  isize dim = H.rows();
  rhs.setZero();
  // stores eigenvector
  rhs.array() += 1. / std::sqrt(dim);
  // stores Hx
  dw.setZero();
  // detail::noalias_symhiv_add(
  // detail::vec_mut(dw), H, detail::vec_mut(rhs));
  dw = H * rhs;
  T eig = 0;
  for (isize i = 0; i < nb_power_iteration; i++) {

    rhs = dw / dw.norm();
    dw.setZero();
    // detail::noalias_symhiv_add(
    // detail::vec_mut(dw), H, detail::vec_mut(rhs));
    dw = H * rhs;
    // calculate associated eigenvalue
    eig = rhs.dot(dw);
    // calculate associated error
    err_v = dw - eig * rhs;
    T err = proxsuite::proxqp::dense::infty_norm(err_v);
    // std::cout << "power iteration max: i " << i << " err " << err <<
    // std::endl;
    if (err <= power_iteration_accuracy) {
      break;
    }
  }
  return eig;
}
template<typename T, typename I>
T
min_eigen_value_via_modified_power_iteration(SparseMat<T, I>& H,
                                             sparse::Vec<T>& dw,
                                             sparse::Vec<T>& rhs,
                                             sparse::Vec<T>& err_v,
                                             T max_eigen_value,
                                             T power_iteration_accuracy,
                                             isize nb_power_iteration)
{
  // performs power iteration on the matrix: max_eigen_value I - H
  // estimates then the minimal eigenvalue with: minimal_eigenvalue =
  // max_eigen_value - eig
  isize dim = H.rows();
  rhs.setZero();
  // stores eigenvector
  rhs.array() += 1. / std::sqrt(dim);
  // stores Hx
  dw.setZero();
  // does not work below with full symmetry...
  // detail::noalias_symhiv_add(
  // detail::vec_mut(dw), H, detail::vec_mut(rhs));
  dw.noalias() = H * rhs;
  dw.array() *= (-1.);
  dw += max_eigen_value * rhs;
  T eig = 0;
  for (isize i = 0; i < nb_power_iteration; i++) {

    rhs = dw / dw.norm();
    dw.setZero();
    // idem
    // detail::noalias_symhiv_add(
    // detail::vec_mut(dw), H, detail::vec_mut(rhs));
    dw.noalias() = H * rhs;
    dw.array() *= (-1.);
    dw += max_eigen_value * rhs;
    // calculate associated eigenvalue
    eig = rhs.dot(dw);
    // calculate associated error
    err_v = dw - eig * rhs;
    T err = proxsuite::proxqp::dense::infty_norm(err_v);
    // std::cout << "power iteration min: i " << i << " err " << err <<
    // std::endl;
    if (err <= power_iteration_accuracy) {
      break;
    }
  }
  T minimal_eigenvalue = max_eigen_value - eig;
  return minimal_eigenvalue;
}
/////// SETUP ////////
/*!
 * Estimate minimal eigenvalue of a symmetric Matrix via power iteration
 * @param H symmetric matrix.
 * @param power_iteration_accuracy power iteration algorithm accuracy tracked
 * @param nb_power_iteration maximal number of power iteration executed
 *
 */
template<typename T, typename I>
T
estimate_minimal_eigen_value_of_symmetric_matrix(SparseMat<T, I>& H,
                                                 T power_iteration_accuracy,
                                                 isize nb_power_iteration)
{
  PROXSUITE_THROW_PRETTY(
    (!H.isApprox(H.transpose(), std::numeric_limits<T>::epsilon())),
    std::invalid_argument,
    "H is not symmetric.");
  PROXSUITE_CHECK_ARGUMENT_SIZE(
    H.rows(),
    H.cols(),
    "H has a number of rows different of the number of columns.");
  isize dim = H.rows();
  T res(0.);
  sparse::Vec<T> dw(dim);
  sparse::Vec<T> rhs(dim);
  sparse::Vec<T> err(dim);
  T dominant_eigen_value = power_iteration<T, I>(
    H, dw, rhs, err, power_iteration_accuracy, nb_power_iteration);
  T min_eigenvalue =
    min_eigen_value_via_modified_power_iteration<T, I>(H,
                                                       dw,
                                                       rhs,
                                                       err,
                                                       dominant_eigen_value,
                                                       power_iteration_accuracy,
                                                       nb_power_iteration);
  // std::cout << "dominant_eigen_value " << dominant_eigen_value<< "
  // min_eigenvalue " << min_eigenvalue << std::endl;
  res = std::min(min_eigenvalue, dominant_eigen_value);
  return res;
}
/////// SETUP ////////
/*!
 * Estimate H minimal eigenvalue
 * @param settings solver settings
 * @param results solver results.
 * @param manual_minimal_H_eigenvalue minimal H eigenvalue estimate.
 */
template<typename T>
void
update_default_rho_with_minimal_Hessian_eigen_value(
  optional<T> manual_minimal_H_eigenvalue,
  Results<T>& results,
  Settings<T>& settings)
{
  if (manual_minimal_H_eigenvalue != nullopt) {
    settings.default_H_eigenvalue_estimate =
      manual_minimal_H_eigenvalue.value();
    results.info.minimal_H_eigenvalue_estimate =
      settings.default_H_eigenvalue_estimate;
  }
  settings.default_rho += std::abs(results.info.minimal_H_eigenvalue_estimate);
  results.info.rho = settings.default_rho;
}
/*!
 * Update the proximal parameters of the results object.
 *
 * @param rho_new primal proximal parameter
 * @param mu_eq_new dual equality proximal parameter
 * @param mu_in_new dual inequality proximal parameter
 * @param results solver result
 */
template<typename T, typename I>
void
update_proximal_parameters(Settings<T>& settings,
                           Results<T>& results,
                           Workspace<T, I>& work,
                           optional<T> rho_new,
                           optional<T> mu_eq_new,
                           optional<T> mu_in_new)
{
  if (rho_new != nullopt) {
    settings.default_rho = rho_new.value();
    results.info.rho = rho_new.value();
    work.internal.proximal_parameter_update = true;
  }
  if (mu_eq_new != nullopt) {
    settings.default_mu_eq = mu_eq_new.value();
    results.info.mu_eq = mu_eq_new.value();
    results.info.mu_eq_inv = T(1) / results.info.mu_eq;
    work.internal.proximal_parameter_update = true;
  }
  if (mu_in_new != nullopt) {
    settings.default_mu_in = mu_in_new.value();
    results.info.mu_in = mu_in_new.value();
    results.info.mu_in_inv = T(1) / results.info.mu_in;
    work.internal.proximal_parameter_update = true;
  }
}
/*!
 * Warm start the results primal and dual variables.
 *
 * @param x_wm primal proximal parameter
 * @param y_wm dual equality proximal parameter
 * @param z_wm dual inequality proximal parameter
 * @param results solver result
 * @param settings solver settings
 */
template<typename T, typename I>
void
warm_start(optional<VecRef<T>> x_wm,
           optional<VecRef<T>> y_wm,
           optional<VecRef<T>> z_wm,
           Results<T>& results,
           Settings<T>& settings,
           Model<T, I>& model)
{
  if (x_wm == nullopt && y_wm == nullopt && z_wm == nullopt)
    return;

  settings.initial_guess = InitialGuessStatus::WARM_START;

  // first check problem dimensions
  if (x_wm != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      x_wm.value().rows(),
      model.dim,
      "the dimension wrt primal variable x for warm start is not valid.");
  }

  if (y_wm != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(y_wm.value().rows(),
                                  model.n_eq,
                                  "the dimension wrt equality constrained "
                                  "variables for warm start is not valid.");
  }

  if (z_wm != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      z_wm.value().rows(),
      model.n_in,
      "the dimension wrt inequality constrained variables for warm start "
      "is not valid.");
  }

  if (x_wm != nullopt) {
    results.x = x_wm.value().eval();
  }

  if (y_wm != nullopt) {
    results.y = y_wm.value().eval();
  }

  if (z_wm != nullopt) {
    results.z = z_wm.value().eval();
  }
}

/*!
 * Setups the QP solver model.
 *
 * @param qp view of the QP model.
 * @param work solver workspace.
 * @param settings solver settings.
 * @param data solver model.
 * @param results solver result.
 * @param precond preconditioner.
 * @param preconditioner_status bool variable for deciding whether executing the
 * preconditioning algorithm, or keeping previous preconditioning variables, or
 * using the identity preconditioner (i.e., no preconditioner).
 */
template<typename T, typename I, typename P>
void
qp_setup(QpView<T, I> qp,
         Results<T>& results,
         Model<T, I>& data,
         Workspace<T, I>& work,
         Settings<T>& settings,
         P& precond,
         PreconditionerStatus& preconditioner_status)
{
  isize n = qp.H.nrows();
  isize n_eq = qp.AT.ncols();
  isize n_in = qp.CT.ncols();

  if (results.x.rows() != n) {
    results.x.resize(n);
    results.x.setZero();
  }
  if (results.y.rows() != n_eq) {
    results.y.resize(n_eq);
    results.y.setZero();
  }
  if (results.z.rows() != n_in) {
    results.z.resize(n_in);
    results.z.setZero();
  }
  if (work.active_inequalities.len() != n_in) {
    work.active_inequalities.resize(n_in);
    for (isize i = 0; i < n_in; ++i) {
      work.active_inequalities[i] = false;
    }
  }
  if (work.active_set_up.rows() != n_in) {
    work.active_set_up.resize(n_in);
    for (isize i = 0; i < n_in; ++i) {
      work.active_set_up[i] = false;
    }
  }
  if (work.active_set_low.rows() != n_in) {
    work.active_set_low.resize(n_in);
    for (isize i = 0; i < n_in; ++i) {
      work.active_set_low[i] = false;
    }
  }
  bool execute_preconditioner_or_not = false;
  switch (preconditioner_status) {
    case PreconditionerStatus::EXECUTE:
      execute_preconditioner_or_not = true;
      break;
    case PreconditionerStatus::IDENTITY:
      execute_preconditioner_or_not = false;
      break;
    case PreconditionerStatus::KEEP:
      // keep previous one
      execute_preconditioner_or_not = false;
      break;
  }
  // performs scaling according to options chosen + stored model value
  work.setup_impl(
    qp,
    data,
    settings,
    execute_preconditioner_or_not,
    precond,
    P::scale_qp_in_place_req(proxsuite::linalg::veg::Tag<T>{}, n, n_eq, n_in));
  switch (settings.initial_guess) { // the following is used when initiliazing
                                    // the Qp object or updating it
    case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {

      if (work.internal.proximal_parameter_update) {
        results.cleanup_all_except_prox_parameters();
      } else {
        results.cleanup(settings);
      }
      break;
    }
    case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
      // keep solutions but restart workspace and results

      if (work.internal.proximal_parameter_update) {
        results.cleanup_statistics();
      } else {
        results.cold_start(settings);
      }
      break;
    }
    case InitialGuessStatus::NO_INITIAL_GUESS: {

      if (work.internal.proximal_parameter_update) {
        results.cleanup_all_except_prox_parameters();
      } else {
        results.cleanup(settings);
      }
      break;
    }
    case InitialGuessStatus::WARM_START: {

      if (work.internal.proximal_parameter_update) {
        results.cleanup_all_except_prox_parameters();
      } else {
        results.cleanup(settings);
      }
      break;
    }
    case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
      // keep workspace and results solutions except statistics

      results.cleanup_statistics(); // always keep prox parameters (changed or
                                    // previous ones)
      break;
    }
  }
  // if user chose Automatic as sparse backend, store in results which backend
  // of SparseCholesky or MatrixFree had been used
  if (settings.sparse_backend == SparseBackend::Automatic) {
    if (work.internal.do_ldlt) {
      results.info.sparse_backend = SparseBackend::SparseCholesky;
    } else {
      results.info.sparse_backend = SparseBackend::MatrixFree;
    }
  }
  // if user selected a specfic sparse backend, store it in results
  else {
    results.info.sparse_backend = settings.sparse_backend;
  }
}
/*!
 * Checks whether matrix b has the same sparsity structure as matrix a.
 *
 * @param a matrix.
 * @param b matrix.
 */
template<typename T, typename I>
auto
have_same_structure(proxsuite::linalg::sparse::MatRef<T, I> a,
                    proxsuite::linalg::sparse::MatRef<T, I> b) -> bool
{
  if (a.nrows() != b.nrows())
    return false;
  if (a.ncols() != b.ncols())
    return false;
  for (usize j = 0; j < static_cast<usize>(a.ncols()); ++j) {
    usize n_elems(a.col_end(j) - a.col_start(j));
    usize n_elems_to_compare(b.col_end(j) - b.col_start(j));
    if (n_elems != n_elems_to_compare)
      return false;
    for (usize p = 0; p < n_elems; ++p) {
      isize i_a = a.row_indices()[a.col_start(j) + p];
      isize i_b = b.row_indices()[b.col_start(j) + p];
      if (i_a != i_b)
        return false;
    }
  }
  return true;
}
/*!
 * Copies matrix b elements into matrix a.
 *
 * @param a matrix.
 * @param b matrix.
 */
template<typename T, typename I>
void
copy(proxsuite::linalg::sparse::MatMut<T, I> a,
     proxsuite::linalg::sparse::MatRef<T, I> b)
{
  // assume same sparsity structure for a and b
  // copy b into a
  for (usize j = 0; j < static_cast<usize>(a.ncols()); ++j) {
    auto a_start = a.values_mut() + a.col_start(j);
    auto b_start = b.values() + b.col_start(j);

    usize n_elems = static_cast<usize>(a.col_end(j) - a.col_start(j));

    for (usize p = 0; p < n_elems; ++p) {
      a_start[p] = b_start[p];
    }
  }
}

} // namespace sparse
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_SPARSE_HELPERS_HPP */
