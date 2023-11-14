//
// Copyright (c) 2022-2023 INRIA
//
/**
 * @file helpers.hpp
 */

#ifndef PROXSUITE_PROXQP_DENSE_HELPERS_HPP
#define PROXSUITE_PROXQP_DENSE_HELPERS_HPP

#include <proxsuite/proxqp/results.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/proxqp/dense/fwd.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>
#include <chrono>
#include <proxsuite/helpers/optional.hpp>
#include <Eigen/Eigenvalues>

namespace proxsuite {
namespace proxqp {
namespace dense {

template<typename T,
         typename MatIn,
         typename VecIn1,
         typename VecIn2,
         typename VecIn3>
T
power_iteration(const Eigen::MatrixBase<MatIn>& H,
                const Eigen::MatrixBase<VecIn1>& dw,
                const Eigen::MatrixBase<VecIn2>& rhs,
                const Eigen::MatrixBase<VecIn3>& err_v,
                T power_iteration_accuracy,
                isize nb_power_iteration)
{
  auto& dw_cc = dw.const_cast_derived();
  auto& rhs_cc = rhs.const_cast_derived();
  auto& err_v_cc = err_v.const_cast_derived();
  // computes maximal eigen value of the bottom right matrix of the LDLT
  isize dim = H.rows();
  rhs_cc.setZero();
  // stores eigenvector
  rhs_cc.array() += 1. / std::sqrt(dim);
  // stores Hx
  dw_cc.noalias() = H.template selfadjointView<Eigen::Lower>() * rhs_cc; // Hx
  T eig = 0;
  for (isize i = 0; i < nb_power_iteration; i++) {

    rhs_cc = dw_cc / dw_cc.norm();
    dw_cc.noalias() = (H.template selfadjointView<Eigen::Lower>() * rhs_cc);
    // calculate associated eigenvalue
    eig = rhs.dot(dw_cc);
    // calculate associated error
    err_v_cc = dw_cc - eig * rhs_cc;
    T err = proxsuite::proxqp::dense::infty_norm(err_v_cc);
    // std::cout << "power iteration max: i " << i << " err " << err <<
    // std::endl;
    if (err <= power_iteration_accuracy) {
      break;
    }
  }
  return eig;
}
template<typename T,
         typename MatIn,
         typename VecIn1,
         typename VecIn2,
         typename VecIn3>
T
min_eigen_value_via_modified_power_iteration(
  const Eigen::MatrixBase<MatIn>& H,
  const Eigen::MatrixBase<VecIn1>& dw,
  const Eigen::MatrixBase<VecIn2>& rhs,
  const Eigen::MatrixBase<VecIn3>& err_v,
  T max_eigen_value,
  T power_iteration_accuracy,
  isize nb_power_iteration)
{
  // performs power iteration on the matrix: max_eigen_value I - H
  // estimates then the minimal eigenvalue with: minimal_eigenvalue =
  // max_eigen_value - eig
  auto& dw_cc = dw.const_cast_derived();
  auto& rhs_cc = rhs.const_cast_derived();
  auto& err_v_cc = err_v.const_cast_derived();
  isize dim = H.rows();
  rhs_cc.setZero();
  // stores eigenvector
  rhs_cc.array() += 1. / std::sqrt(dim);
  // stores Hx
  dw_cc.noalias() =
    -(H.template selfadjointView<Eigen::Lower>() * rhs_cc); // Hx
  dw_cc += max_eigen_value * rhs_cc;
  T eig = 0;
  for (isize i = 0; i < nb_power_iteration; i++) {

    rhs_cc = dw_cc / dw_cc.norm();
    dw_cc.noalias() = -(H.template selfadjointView<Eigen::Lower>() * rhs_cc);
    dw_cc += max_eigen_value * rhs_cc;
    // calculate associated eigenvalue
    eig = rhs_cc.dot(dw_cc);
    // calculate associated error
    err_v_cc = dw_cc - eig * rhs_cc;
    T err = proxsuite::proxqp::dense::infty_norm(err_v_cc);
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
 * Estimate minimal eigenvalue of a symmetric Matrix
 * @param H symmetric matrix.
 * @param EigenValueEstimateMethodOption
 * @param power_iteration_accuracy power iteration algorithm accuracy tracked
 * @param nb_power_iteration maximal number of power iteration executed
 *
 */
template<typename T, typename MatIn>
T
estimate_minimal_eigen_value_of_symmetric_matrix(
  const Eigen::MatrixBase<MatIn>& H,
  EigenValueEstimateMethodOption estimate_method_option,
  T power_iteration_accuracy,
  isize nb_power_iteration)
{
  PROXSUITE_THROW_PRETTY(
    (!H.isApprox(H.transpose(), std::numeric_limits<T>::epsilon())),
    std::invalid_argument,
    "H is not symmetric.");
  if (H.size()) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      H.rows(),
      H.cols(),
      "H has a number of rows different of the number of columns.");
  }
  isize dim = H.rows();
  T res(0.);
  switch (estimate_method_option) {
    case EigenValueEstimateMethodOption::PowerIteration: {
      Vec<T> dw(dim);
      Vec<T> rhs(dim);
      Vec<T> err(dim);
      T dominant_eigen_value = power_iteration(
        H, dw, rhs, err, power_iteration_accuracy, nb_power_iteration);
      T min_eigenvalue =
        min_eigen_value_via_modified_power_iteration(H,
                                                     dw,
                                                     rhs,
                                                     err,
                                                     dominant_eigen_value,
                                                     power_iteration_accuracy,
                                                     nb_power_iteration);
      res = std::min(min_eigenvalue, dominant_eigen_value);
    } break;
    case EigenValueEstimateMethodOption::ExactMethod: {
      Eigen::SelfAdjointEigenSolver<Mat<T>> es(H, Eigen::EigenvaluesOnly);
      res = T(es.eigenvalues()[0]);
    } break;
  }
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
 * Computes the equality constrained initial guess of a QP problem.
 *
 * @param qpwork workspace of the solver.
 * @param qpsettings settings of the solver.
 * @param qpmodel QP problem as defined by the user (without any scaling
 * performed).
 * @param qpresults solution results.
 */
template<typename T>
void
compute_equality_constrained_initial_guess(Workspace<T>& qpwork,
                                           const Settings<T>& qpsettings,
                                           const Model<T>& qpmodel,
                                           const isize n_constraints,
                                           const DenseBackend& dense_backend,
                                           const HessianType& hessian_type,
                                           Results<T>& qpresults)
{

  qpwork.rhs.setZero();
  qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;
  iterative_solve_with_permut_fact( //
    qpsettings,
    qpmodel,
    qpresults,
    qpwork,
    n_constraints,
    dense_backend,
    hessian_type,
    T(1),
    qpmodel.dim + qpmodel.n_eq);

  qpresults.x = qpwork.dw_aug.head(qpmodel.dim);
  qpresults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
  qpwork.dw_aug.setZero();
  qpwork.rhs.setZero();
}

/*!
 * Setups and performs the first factorization of the regularized KKT matrix of
 * the problem.
 *
 * @param qpwork workspace of the solver.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solution results.
 */
template<typename T>
void
setup_factorization(Workspace<T>& qpwork,
                    const Model<T>& qpmodel,
                    Results<T>& qpresults,
                    const DenseBackend& dense_backend,
                    const HessianType& hessian_type)
{

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    qpwork.ldl_stack.as_mut(),
  };
  switch (hessian_type) {
    case HessianType::Dense:
      qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
      break;
    case HessianType::Zero:
      qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).setZero();
      break;
    case HessianType::Diagonal:
      qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
      break;
  }
  qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
    qpresults.info.rho;
  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT:
      qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
        qpwork.A_scaled.transpose();
      qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) =
        qpwork.A_scaled;
      qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
      qpwork.kkt.diagonal()
        .segment(qpmodel.dim, qpmodel.n_eq)
        .setConstant(-qpresults.info.mu_eq);
      qpwork.ldl.factorize(qpwork.kkt.transpose(), stack);
      break;
    case DenseBackend::PrimalLDLT:
      qpwork.kkt.noalias() += qpresults.info.mu_eq_inv *
                              (qpwork.A_scaled.transpose() * qpwork.A_scaled);
      qpwork.ldl.factorize(qpwork.kkt.transpose(), stack);
      break;
    case DenseBackend::Automatic:
      break;
  }
}
/*!
 * Performs the equilibration of the QP problem for reducing its
 * ill-conditionness.
 *
 * @param qpwork workspace of the solver.
 * @param qpsettings settings of the solver.
 * @param ruiz ruiz preconditioner.
 * @param execute_preconditioner boolean variable for executing or not the ruiz
 * preconditioner. If set to False, it uses the previous preconditioning
 * variables (initialized to the identity preconditioner if it is the first
 * scaling performed).
 */
template<typename T>
void
setup_equilibration(Workspace<T>& qpwork,
                    const Settings<T>& qpsettings,
                    const bool box_constraints,
                    const HessianType hessian_type,
                    preconditioner::RuizEquilibration<T>& ruiz,
                    bool execute_preconditioner)
{

  QpViewBoxMut<T> qp_scaled{
    { from_eigen, qpwork.H_scaled },     { from_eigen, qpwork.g_scaled },
    { from_eigen, qpwork.A_scaled },     { from_eigen, qpwork.b_scaled },
    { from_eigen, qpwork.C_scaled },     { from_eigen, qpwork.u_scaled },
    { from_eigen, qpwork.l_scaled },     { from_eigen, qpwork.i_scaled },
    { from_eigen, qpwork.l_box_scaled }, { from_eigen, qpwork.u_box_scaled },
  };

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    qpwork.ldl_stack.as_mut(),
  };
  ruiz.scale_qp_in_place(qp_scaled,
                         execute_preconditioner,
                         qpsettings.primal_infeasibility_solving,
                         qpsettings.preconditioner_max_iter,
                         qpsettings.preconditioner_accuracy,
                         hessian_type,
                         box_constraints,
                         stack);
  qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);
}

/*!
 * Setups the solver initial guess.
 *
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
void
initial_guess(Workspace<T>& qpwork,
              Settings<T>& qpsettings,
              Model<T>& qpmodel,
              Results<T>& qpresults)
{

  switch (qpsettings.initial_guess) {
    case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
      compute_equality_constrained_initial_guess(
        qpwork, qpsettings, qpmodel, qpresults);
      break;
    }
  }
}
/*!
 * Updates the QP solver model.
 *
 * @param H quadratic cost input defining the QP model.
 * @param g linear cost input defining the QP model.
 * @param A equality constraint matrix input defining the QP model.
 * @param b equality constraint vector input defining the QP model.
 * @param C inequality constraint matrix input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param u upper inequality constraint vector input defining the QP model.
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpmodel solver model.
 * @param qpresults solver result.
 */

template<typename T>
void
update(optional<MatRef<T>> H,
       optional<VecRef<T>> g,
       optional<MatRef<T>> A,
       optional<VecRef<T>> b,
       optional<MatRef<T>> C,
       optional<VecRef<T>> l,
       optional<VecRef<T>> u,
       optional<VecRef<T>> l_box,
       optional<VecRef<T>> u_box,
       Model<T>& model,
       Workspace<T>& work,
       const bool box_constraints)
{
  // check the model is valid
  if (g != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(g.value().size(),
                                  model.dim,
                                  "the dimension wrt the primal variable x "
                                  "variable for updating g is not valid.");
  }
  if (b != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(b.value().size(),
                                  model.n_eq,
                                  "the dimension wrt equality constrained "
                                  "variables for updating b is not valid.");
  }
  if (u != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(u.value().size(),
                                  model.n_in,
                                  "the dimension wrt inequality constrained "
                                  "variables for updating u is not valid.");
  }
  if (l != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(l.value().size(),
                                  model.n_in,
                                  "the dimension wrt inequality constrained "
                                  "variables for updating l is not valid.");
  }
  if (H != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      H.value().rows(),
      model.dim,
      "the row dimension for updating H is not valid.");
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      H.value().cols(),
      model.dim,
      "the column dimension for updating H is not valid.");
  }
  if (A != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      A.value().rows(),
      model.n_eq,
      "the row dimension for updating A is not valid.");
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      A.value().cols(),
      model.dim,
      "the column dimension for updating A is not valid.");
  }
  if (C != nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      C.value().rows(),
      model.n_in,
      "the row dimension for updating C is not valid.");
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      C.value().cols(),
      model.dim,
      "the column dimension for updating C is not valid.");
  }

  // update the model
  if (g != nullopt) {
    model.g = g.value().eval();
  }
  if (b != nullopt) {
    model.b = b.value().eval();
  }
  if (u != nullopt) {
    model.u = u.value().eval();
  }
  if (l != nullopt) {
    model.l = l.value().eval();
  }
  if (u_box != nullopt && box_constraints) {
    model.u_box = u_box.value();
  } // else qpmodel.u_box remains initialized to a matrix with zero elements or
    // zero shape

  if (l_box != nullopt && box_constraints) {
    model.l_box = l_box.value();
  } // else qpmodel.l_box remains initialized to a matrix with zero elements or
    // zero shape

  if (H != nullopt || A != nullopt || C != nullopt) {
    work.refactorize = true;
  }

  if (H != nullopt) {
    model.H = H.value();
  }
  if (A != nullopt) {
    model.A = A.value();
  }
  if (C != nullopt) {
    model.C = C.value();
  }
  assert(model.is_valid(box_constraints));
}
/*!
 * Setups the QP solver model.
 *
 * @param H quadratic cost input defining the QP model.
 * @param g linear cost input defining the QP model.
 * @param A equality constraint matrix input defining the QP model.
 * @param b equality constraint vector input defining the QP model.
 * @param C inequality constraint matrix input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param u upper inequality constraint vector input defining the QP model.
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpmodel solver model.
 * @param qpresults solver result.
 * @param ruiz ruiz preconditioner.
 * @param preconditioner_status bool variable for deciding whether executing the
 * preconditioning algorithm, or keeping previous preconditioning variables, or
 * using the identity preconditioner (i.e., no preconditioner).
 */
template<typename T>
void
setup( //
  optional<MatRef<T>> H,
  optional<VecRef<T>> g,
  optional<MatRef<T>> A,
  optional<VecRef<T>> b,
  optional<MatRef<T>> C,
  optional<VecRef<T>> l,
  optional<VecRef<T>> u,
  optional<VecRef<T>> l_box,
  optional<VecRef<T>> u_box,
  Settings<T>& qpsettings,
  Model<T>& qpmodel,
  Workspace<T>& qpwork,
  Results<T>& qpresults,
  const bool box_constraints,
  preconditioner::RuizEquilibration<T>& ruiz,
  PreconditionerStatus preconditioner_status,
  const HessianType hessian_type)
{

  switch (qpsettings.initial_guess) {
    case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
      if (qpwork.proximal_parameter_update) {
        qpresults.cleanup_all_except_prox_parameters();
      } else {
        qpresults.cleanup(qpsettings);
      }
      qpwork.cleanup(box_constraints);
      break;
    }
    case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
      // keep solutions but restart workspace and results
      if (qpwork.proximal_parameter_update) {
        qpresults.cleanup_statistics();
      } else {
        qpresults.cold_start(qpsettings);
      }
      qpwork.cleanup(box_constraints);
      break;
    }
    case InitialGuessStatus::NO_INITIAL_GUESS: {
      if (qpwork.proximal_parameter_update) {
        qpresults.cleanup_all_except_prox_parameters();
      } else {
        qpresults.cleanup(qpsettings);
      }
      qpwork.cleanup(box_constraints);
      break;
    }
    case InitialGuessStatus::WARM_START: {
      if (qpwork.proximal_parameter_update) {
        qpresults
          .cleanup_all_except_prox_parameters(); // the warm start is given at
                                                 // the solve function
      } else {
        qpresults.cleanup(qpsettings);
      }
      qpwork.cleanup(box_constraints);
      break;
    }
    case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
      if (qpwork.refactorize || qpwork.proximal_parameter_update) {
        qpwork.cleanup(box_constraints); // meaningful for when there is an
                                         // upate of the model and one wants to
                                         // warm start with previous result
        qpwork.refactorize = true;
      }
      qpresults.cleanup_statistics();
      break;
    }
  }
  if (H != nullopt) {
    qpmodel.H = H.value();
  } // else qpmodel.H remains initialzed to a matrix with zero elements
  if (g != nullopt) {
    qpmodel.g = g.value();
  }

  if (A != nullopt) {
    qpmodel.A = A.value();
  } // else qpmodel.A remains initialized to a matrix with zero elements or zero
    // shape

  if (b != nullopt) {
    qpmodel.b = b.value();
  } // else qpmodel.b remains initialized to a matrix with zero elements or zero
    // shape

  if (C != nullopt) {
    qpmodel.C = C.value();
  } // else qpmodel.C remains initialized to a matrix with zero elements or zero
    // shape

  if (u != nullopt) {
    qpmodel.u = u.value();
  } // else qpmodel.u remains initialized to a matrix with zero elements or zero
    // shape

  if (l != nullopt) {
    qpmodel.l = l.value();
  } // else qpmodel.l remains initialized to a matrix with zero elements or zero
    // shape
  if (u_box != nullopt) {
    qpmodel.u_box = u_box.value();
  } // else qpmodel.u_box remains initialized to a matrix with zero elements or
    // zero shape

  if (l_box != nullopt) {
    qpmodel.l_box = l_box.value();
  } // else qpmodel.l_box remains initialized to a matrix with zero elements or
    // zero shape
  assert(qpmodel.is_valid(box_constraints));
  switch (hessian_type) {
    case HessianType::Zero:
      break;
    case HessianType::Dense:
      qpwork.H_scaled = qpmodel.H;
      break;
    case HessianType::Diagonal:
      qpwork.H_scaled = qpmodel.H;
      break;
  }
  qpwork.g_scaled = qpmodel.g;
  qpwork.A_scaled = qpmodel.A;
  qpwork.b_scaled = qpmodel.b;
  qpwork.C_scaled = qpmodel.C;
  qpwork.u_scaled =
    (qpmodel.u.array() <= T(1.E20))
      .select(qpmodel.u,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in).array() +
                T(1.E20));
  qpwork.l_scaled =
    (qpmodel.l.array() >= T(-1.E20))
      .select(qpmodel.l,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in).array() -
                T(1.E20));
  if (box_constraints) {
    qpwork.u_box_scaled =
      (qpmodel.u_box.array() <= T(1.E20))
        .select(qpmodel.u_box,
                Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.dim).array() +
                  T(1.E20));
    qpwork.l_box_scaled =
      (qpmodel.l_box.array() >= T(-1.E20))
        .select(qpmodel.l_box,
                Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.dim).array() -
                  T(1.E20));
  }

  qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);
  switch (preconditioner_status) {
    case PreconditionerStatus::EXECUTE:
      setup_equilibration(
        qpwork, qpsettings, box_constraints, hessian_type, ruiz, true);
      break;
    case PreconditionerStatus::IDENTITY:
      setup_equilibration(
        qpwork, qpsettings, box_constraints, hessian_type, ruiz, false);
      break;
    case PreconditionerStatus::KEEP:
      // keep previous one
      setup_equilibration(
        qpwork, qpsettings, box_constraints, hessian_type, ruiz, false);
      break;
  }
}
////// UPDATES ///////

/*!
 * Update the proximal parameters of the results object.
 *
 * @param rho_new primal proximal parameter.
 * @param mu_eq_new dual equality proximal parameter.
 * @param mu_in_new dual inequality proximal parameter.
 * @param results solver results.
 */
template<typename T>
void
update_proximal_parameters(Settings<T>& settings,
                           Results<T>& results,
                           Workspace<T>& work,
                           optional<T> rho_new,
                           optional<T> mu_eq_new,
                           optional<T> mu_in_new)
{

  if (rho_new != nullopt) {
    settings.default_rho = rho_new.value();
    results.info.rho = rho_new.value();
    work.proximal_parameter_update = true;
  }
  if (mu_eq_new != nullopt) {
    settings.default_mu_eq = mu_eq_new.value();
    results.info.mu_eq = mu_eq_new.value();
    results.info.mu_eq_inv = T(1) / results.info.mu_eq;
    work.proximal_parameter_update = true;
  }
  if (mu_in_new != nullopt) {
    settings.default_mu_in = mu_in_new.value();
    results.info.mu_in = mu_in_new.value();
    results.info.mu_in_inv = T(1) / results.info.mu_in;
    work.proximal_parameter_update = true;
  }
}
/*!
 * Warm start the primal and dual variables.
 *
 * @param x_wm primal warm start.
 * @param y_wm dual equality warm start.
 * @param z_wm dual inequality warm start.
 * @param results solver result.
 * @param settings solver settings.
 */
template<typename T>
void
warm_start(optional<VecRef<T>> x_wm,
           optional<VecRef<T>> y_wm,
           optional<VecRef<T>> z_wm,
           Results<T>& results,
           Settings<T>& settings,
           Model<T>& model)
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
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_HELPERS_HPP */
