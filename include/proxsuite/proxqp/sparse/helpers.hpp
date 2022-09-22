//
// Copyright (c) 2022 INRIA
//
/** \file */
#ifndef PROXSUITE_QP_SPARSE_HELPERS_HPP
#define PROXSUITE_QP_SPARSE_HELPERS_HPP

#include <Eigen/Sparse>
#include <optional>

#include <proxsuite/linalg/veg/vec.hpp>
#include <proxsuite/proxqp/sparse/fwd.hpp>

namespace proxsuite {
namespace proxqp {
namespace sparse {

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
                           std::optional<T> rho_new,
                           std::optional<T> mu_eq_new,
                           std::optional<T> mu_in_new)
{
  if (rho_new != std::nullopt) {
    settings.default_rho = rho_new.value();
    results.info.rho = rho_new.value();
    work.internal.proximal_parameter_update = true;
  }
  if (mu_eq_new != std::nullopt) {
    settings.default_mu_eq = mu_eq_new.value();
    results.info.mu_eq = mu_eq_new.value();
    results.info.mu_eq_inv = T(1) / results.info.mu_eq;
    work.internal.proximal_parameter_update = true;
  }
  if (mu_in_new != std::nullopt) {
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
warm_start(std::optional<VecRef<T>> x_wm,
           std::optional<VecRef<T>> y_wm,
           std::optional<VecRef<T>> z_wm,
           Results<T>& results,
           Settings<T>& settings,
           Model<T, I>& model)
{

  isize n_eq = results.y.rows();
  isize n_in = results.z.rows();
  if (n_eq != 0) {
    if (n_in != 0) {
      if (x_wm != std::nullopt && y_wm != std::nullopt &&
          z_wm != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(
          z_wm.value().rows(),
          model.n_in,
          "the dimension wrt inequality constrained variables for warm start "
          "is not valid.");
        PROXSUITE_CHECK_ARGUMENT_SIZE(y_wm.value().rows(),
                                      model.n_eq,
                                      "the dimension wrt equality constrained "
                                      "variables for warm start is not valid.");
        PROXSUITE_CHECK_ARGUMENT_SIZE(x_wm.value().rows(),
                                      model.dim,
                                      "the dimension wrt primal variable x "
                                      "variable for warm start is not valid.");
        results.x = x_wm.value().eval();
        results.y = y_wm.value().eval();
        results.z = z_wm.value().eval();
        settings.initial_guess = InitialGuessStatus::WARM_START;
      }
    } else {
      // n_in= 0
      if (x_wm != std::nullopt && y_wm != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(y_wm.value().rows(),
                                      model.n_eq,
                                      "the dimension wrt equality constrained "
                                      "variables for warm start is not valid.");
        PROXSUITE_CHECK_ARGUMENT_SIZE(x_wm.value().rows(),
                                      model.dim,
                                      "the dimension wrt primal variable x "
                                      "variable for warm start is not valid.");
        results.x = x_wm.value().eval();
        results.y = y_wm.value().eval();
        settings.initial_guess = InitialGuessStatus::WARM_START;
      }
    }
  } else if (n_in != 0) {
    // n_eq = 0
    if (x_wm != std::nullopt && z_wm != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(z_wm.value().rows(),
                                    model.n_in,
                                    "the dimension wrt inequality constrained "
                                    "variables for warm start is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(x_wm.value().rows(),
                                    model.dim,
                                    "the dimension wrt primal variable x "
                                    "variable for warm start is not valid.");
      results.x = x_wm.value().eval();
      results.z = z_wm.value().eval();
      settings.initial_guess = InitialGuessStatus::WARM_START;
    }
  } else {
    // n_eq = 0 and n_in = 0
    if (x_wm != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(x_wm.value().rows(),
                                    model.dim,
                                    "the dimension wrt primal variable x "
                                    "variable for warm start is not valid.");
      results.x = x_wm.value().eval();
      settings.initial_guess = InitialGuessStatus::WARM_START;
    }
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
  if (results.active_constraints.len() != n_in) {
    results.active_constraints.resize(n_in);
    for (isize i = 0; i < n_in; ++i) {
      results.active_constraints[i] = false;
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

#endif /* end of include guard PROXSUITE_QP_SPARSE_HELPERS_HPP */
