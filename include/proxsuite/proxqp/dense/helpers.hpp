//
// Copyright (c) 2022 INRIA
//
/**
 * @file helpers.hpp
 */

#ifndef PROXSUITE_QP_DENSE_HELPERS_HPP
#define PROXSUITE_QP_DENSE_HELPERS_HPP

#include <proxsuite/proxqp/results.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/proxqp/dense/fwd.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>
#include <chrono>
#include <optional>

namespace proxsuite {
namespace proxqp {
namespace dense {

/////// SETUP ////////
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
                    Results<T>& qpresults)
{

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    qpwork.ldl_stack.as_mut(),
  };

  qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
  qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
    qpresults.info.rho;
  qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
    qpwork.A_scaled.transpose();
  qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
  qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
  qpwork.kkt.diagonal()
    .segment(qpmodel.dim, qpmodel.n_eq)
    .setConstant(-qpresults.info.mu_eq);

  qpwork.ldl.factorize(qpwork.kkt, stack);
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
                    preconditioner::RuizEquilibration<T>& ruiz,
                    bool execute_preconditioner)
{

  QpViewBoxMut<T> qp_scaled{
    { from_eigen, qpwork.H_scaled }, { from_eigen, qpwork.g_scaled },
    { from_eigen, qpwork.A_scaled }, { from_eigen, qpwork.b_scaled },
    { from_eigen, qpwork.C_scaled }, { from_eigen, qpwork.u_scaled },
    { from_eigen, qpwork.l_scaled }
  };

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    qpwork.ldl_stack.as_mut(),
  };
  ruiz.scale_qp_in_place(qp_scaled,
                         execute_preconditioner,
                         qpsettings.preconditioner_max_iter,
                         qpsettings.preconditioner_accuracy,
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
 * @param u upper inequality constraint vector input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpmodel solver model.
 * @param qpresults solver result.
 */

template<typename Mat, typename T>
void
update(std::optional<Mat> H_,
       std::optional<Vec<T>> g_,
       std::optional<Mat> A_,
       std::optional<Vec<T>> b_,
       std::optional<Mat> C_,
       std::optional<Vec<T>> u_,
       std::optional<Vec<T>> l_,
       Model<T>& model,
       Workspace<T>& work)
{
  // check the model is valid
  if (g_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(g_.value().rows(),
                                  model.dim,
                                  "the dimension wrt the primal variable x "
                                  "variable for updating g is not valid.");
  }
  if (b_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(b_.value().rows(),
                                  model.n_eq,
                                  "the dimension wrt equality constrained "
                                  "variables for updating b is not valid.");
  }
  if (u_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(u_.value().rows(),
                                  model.n_in,
                                  "the dimension wrt inequality constrained "
                                  "variables for updating u is not valid.");
  }
  if (l_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(l_.value().rows(),
                                  model.n_in,
                                  "the dimension wrt inequality constrained "
                                  "variables for updating l is not valid.");
  }
  if (H_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      H_.value().rows(),
      model.dim,
      "the row dimension for updating H is not valid.");
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      H_.value().cols(),
      model.dim,
      "the column dimension for updating H is not valid.");
  }
  if (A_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      A_.value().rows(),
      model.n_eq,
      "the row dimension for updating A is not valid.");
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      A_.value().cols(),
      model.dim,
      "the column dimension for updating A is not valid.");
  }
  if (C_ != std::nullopt) {
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      C_.value().rows(),
      model.n_in,
      "the row dimension for updating C is not valid.");
    PROXSUITE_CHECK_ARGUMENT_SIZE(
      C_.value().cols(),
      model.dim,
      "the column dimension for updating C is not valid.");
  }
  // update the model
  if (g_ != std::nullopt) {
    model.g = g_.value().eval();
  }
  if (b_ != std::nullopt) {
    model.b = b_.value().eval();
  }
  if (u_ != std::nullopt) {
    model.u = u_.value().eval();
  }
  if (l_ != std::nullopt) {
    model.l = l_.value().eval();
  }
  if (H_ != std::nullopt) {
    work.refactorize = true;
    if (A_ != std::nullopt) {
      if (C_ != std::nullopt) {
        model.H = Eigen::
          Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
            H_.value());
        model.A = Eigen::
          Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
            A_.value());
        model.C = Eigen::
          Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
            C_.value());
      } else {
        model.H = Eigen::
          Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
            H_.value());
        model.A = Eigen::
          Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
            A_.value());
      }
    } else if (C_ != std::nullopt) {
      model.H = Eigen::
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
          H_.value());
      model.C = Eigen::
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
          C_.value());
    } else {
      model.H = Eigen::
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
          H_.value());
    }
  } else if (A_ != std::nullopt) {
    work.refactorize = true;
    if (C_ != std::nullopt) {
      model.A = Eigen::
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
          A_.value());
      model.C = Eigen::
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
          C_.value());
    } else {
      model.A = Eigen::
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
          A_.value());
    }
  } else if (C_ != std::nullopt) {
    work.refactorize = true;
    model.C = Eigen::
      Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
        C_.value());
  }
}
/*!
 * Setups the QP solver model.
 *
 * @param H quadratic cost input defining the QP model.
 * @param g linear cost input defining the QP model.
 * @param A equality constraint matrix input defining the QP model.
 * @param b equality constraint vector input defining the QP model.
 * @param C inequality constraint matrix input defining the QP model.
 * @param u upper inequality constraint vector input defining the QP model.
 * @param l lower inequality constraint vector input defining the QP model.
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpmodel solver model.
 * @param qpresults solver result.
 * @param ruiz ruiz preconditioner.
 * @param preconditioner_status bool variable for deciding whether executing the
 * preconditioning algorithm, or keeping previous preconditioning variables, or
 * using the identity preconditioner (i.e., no preconditioner).
 */
template<typename Mat, typename T>
void
setup( //
  std::optional<Mat> H,
  std::optional<VecRef<T>> g,
  std::optional<Mat> A,
  std::optional<VecRef<T>> b,
  std::optional<Mat> C,
  std::optional<VecRef<T>> u,
  std::optional<VecRef<T>> l,
  Settings<T>& qpsettings,
  Model<T>& qpmodel,
  Workspace<T>& qpwork,
  Results<T>& qpresults,
  preconditioner::RuizEquilibration<T>& ruiz,
  PreconditionerStatus preconditioner_status)
{

  switch (qpsettings.initial_guess) {
    case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
      if (qpwork.proximal_parameter_update) {
        qpresults.cleanup_all_except_prox_parameters();
      } else {
        qpresults.cleanup(qpsettings);
      }
      qpwork.cleanup();
      break;
    }
    case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
      // keep solutions but restart workspace and results
      if (qpwork.proximal_parameter_update) {
        qpresults.cleanup_statistics();
      } else {
        qpresults.cold_start(qpsettings);
      }
      qpwork.cleanup();
      break;
    }
    case InitialGuessStatus::NO_INITIAL_GUESS: {
      if (qpwork.proximal_parameter_update) {
        qpresults.cleanup_all_except_prox_parameters();
      } else {
        qpresults.cleanup(qpsettings);
      }
      qpwork.cleanup();
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
      qpwork.cleanup();
      break;
    }
    case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
      if (qpwork.refactorize || qpwork.proximal_parameter_update) {
        qpwork.cleanup(); // meaningful for when there is an upate of the model
                          // and one wants to warm start with previous result
        qpwork.refactorize = true;
      }
      qpresults.cleanup_statistics();
      break;
    }
  }
  if (H != std::nullopt) {
    qpmodel.H = H.value();
  } // else qpmodel.H remains initialzed to a matrix with zero elements
  if (g != std::nullopt) {
    qpmodel.g = g.value();
  }

  if (A != std::nullopt) {
    qpmodel.A = A.value();
  } // else qpmodel.A remains initialized to a matrix with zero elements or zero
    // shape

  if (b != std::nullopt) {
    qpmodel.b = b.value();
  } // else qpmodel.b remains initialized to a matrix with zero elements or zero
    // shape

  if (C != std::nullopt) {
    qpmodel.C = C.value();
  } // else qpmodel.C remains initialized to a matrix with zero elements or zero
    // shape

  if (u != std::nullopt) {
    qpmodel.u = u.value();
  } // else qpmodel.u remains initialized to a matrix with zero elements or zero
    // shape

  if (l != std::nullopt) {
    qpmodel.l = l.value();
  } // else qpmodel.l remains initialized to a matrix with zero elements or zero
    // shape

  qpwork.H_scaled = qpmodel.H;
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

  qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
  qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpwork.u_scaled);
  qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpwork.l_scaled);
  qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);

  switch (preconditioner_status) {
    case PreconditionerStatus::EXECUTE:
      setup_equilibration(qpwork, qpsettings, ruiz, true);
      break;
    case PreconditionerStatus::IDENTITY:
      setup_equilibration(qpwork, qpsettings, ruiz, false);
      break;
    case PreconditionerStatus::KEEP:
      // keep previous one
      setup_equilibration(qpwork, qpsettings, ruiz, false);
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
                           std::optional<T> rho_new,
                           std::optional<T> mu_eq_new,
                           std::optional<T> mu_in_new)
{

  if (rho_new != std::nullopt) {
    settings.default_rho = rho_new.value();
    results.info.rho = rho_new.value();
    work.proximal_parameter_update = true;
  }
  if (mu_eq_new != std::nullopt) {
    settings.default_mu_eq = mu_eq_new.value();
    results.info.mu_eq = mu_eq_new.value();
    results.info.mu_eq_inv = T(1) / results.info.mu_eq;
    work.proximal_parameter_update = true;
  }
  if (mu_in_new != std::nullopt) {
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
warm_start(std::optional<VecRef<T>> x_wm,
           std::optional<VecRef<T>> y_wm,
           std::optional<VecRef<T>> z_wm,
           Results<T>& results,
           Settings<T>& settings,
           Model<T>& model)
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
        PROXSUITE_CHECK_ARGUMENT_SIZE(
          x_wm.value().rows(),
          model.dim,
          "the dimension wrt primal variable x for warm start is not valid.");
        results.x = x_wm.value().eval();
        results.y = y_wm.value().eval();
        results.z = z_wm.value().eval();
      }
    } else {
      // n_in= 0
      if (x_wm != std::nullopt && y_wm != std::nullopt) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(y_wm.value().rows(),
                                      model.n_eq,
                                      "the dimension wrt equality constrained "
                                      "variables for warm start is not valid.");
        PROXSUITE_CHECK_ARGUMENT_SIZE(
          x_wm.value().rows(),
          model.dim,
          "the dimension wrt primal variable x for warm start is not valid.");
        results.x = x_wm.value().eval();
        results.y = y_wm.value().eval();
      }
    }
  } else if (n_in != 0) {
    // n_eq = 0
    if (x_wm != std::nullopt && z_wm != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(z_wm.value().rows(),
                                    model.n_in,
                                    "the dimension wrt inequality constrained "
                                    "variables for warm start is not valid.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        x_wm.value().rows(),
        model.dim,
        "the dimension wrt primal variable x for warm start is not valid.");
      results.x = x_wm.value().eval();
      results.z = z_wm.value().eval();
    }
  } else {
    // n_eq = 0 and n_in = 0
    if (x_wm != std::nullopt) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
        x_wm.value().rows(),
        model.dim,
        "the dimension wrt primal variable x for warm start is not valid.");
      results.x = x_wm.value().eval();
    }
  }

  settings.initial_guess = InitialGuessStatus::WARM_START;
}
} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_HELPERS_HPP */
