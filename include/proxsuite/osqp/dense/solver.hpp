//
// Copyright (c) 2025 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_SOLVER_HPP
#define PROXSUITE_OSQP_DENSE_SOLVER_HPP

#include <proxsuite/linalg/dense/ldlt.hpp>
#include <proxsuite/linalg/veg/internal/typedefs.hpp>
#include "proxsuite/proxqp/dense/solver.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
#include "proxsuite/proxqp/dense/preconditioner/ruiz.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/proxqp/status.hpp"
#include "proxsuite/solvers/common/utils.hpp"
#include <iostream>

namespace proxsuite {
namespace osqp {
namespace dense {

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::dense;
namespace plv = proxsuite::linalg::veg;

/*!
 * Computes the scaled primal - dual residual ratio to update mu in OSQP.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
T
compute_update_ratio_primal_dual(const Model<T>& qpmodel,
                                 Results<T>& qpresults,
                                 Workspace<T>& qpwork,
                                 const bool box_constraints,
                                 const HessianType hessian_type)
{
  proxsuite::common::global_primal_residual_scaled(
    qpmodel, qpresults, qpwork, box_constraints);
  T norm_primal_residual_scaled = infty_norm(qpwork.primal_residual_scaled);

  proxsuite::common::global_dual_residual_scaled(
    qpresults, qpwork, qpmodel, box_constraints, hessian_type);
  T norm_dual_residual_scaled = infty_norm(qpwork.dual_residual_scaled);

  T epsilon = 1e-10;

  T norm_Ax = infty_norm(qpwork.A_scaled * qpresults.x);
  T norm_Cx = infty_norm(qpwork.C_scaled * qpresults.x);
  if (box_constraints) {
    norm_Cx = std::max(
      norm_Cx,
      infty_norm((qpwork.i_scaled.array() * qpresults.x.array()).matrix()));
  }
  T norm_zeta =
    std::max(infty_norm(qpwork.zeta_eq), infty_norm(qpwork.zeta_in));
  T max_Ax_Cx = std::max(norm_Ax, norm_Cx);
  T max_scale_primal = std::max(max_Ax_Cx, norm_zeta);
  T primal_term = norm_primal_residual_scaled / (max_scale_primal + epsilon);

  T norm_Hx;
  switch (hessian_type) {
    case HessianType::Zero:
      norm_Hx = 0;
      break;
    case HessianType::Dense:
      norm_Hx = infty_norm(
        qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * qpresults.x);
      break;
    case HessianType::Diagonal:
      norm_Hx = infty_norm(
        (qpwork.H_scaled.diagonal().array() * qpresults.x.array()).matrix());
      break;
  }
  T norm_ATy = infty_norm(qpwork.A_scaled.transpose() * qpresults.y);
  T norm_CTz =
    infty_norm(qpwork.C_scaled.transpose() * qpresults.z.head(qpmodel.n_in));
  if (box_constraints) {
    norm_CTz = std::max(norm_Cx,
                        infty_norm((qpwork.i_scaled.array() *
                                    qpresults.z.tail(qpmodel.dim).array())
                                     .matrix()));
  }
  T norm_g = infty_norm(qpwork.g_scaled);
  T max_Hx_g = std::max(norm_Hx, norm_g);
  T max_ATy_CTz = std::max(norm_ATy, norm_CTz);
  T max_scale_dual = std::max(max_Hx_g, max_ATy_CTz);
  T dual_term = norm_dual_residual_scaled / (max_scale_dual + epsilon);

  T update_ratio_primal_dual = std::sqrt(primal_term / (dual_term + epsilon));
  return update_ratio_primal_dual;
}
/*!
 * Updates the proximal parameters mu_eq and mu_in in the OSQP algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
update_mu(const Settings<T>& qpsettings,
          const Model<T>& qpmodel,
          Results<T>& qpresults,
          Workspace<T>& qpwork,
          const bool box_constraints,
          const isize n_constraints,
          const DenseBackend dense_backend,
          const HessianType hessian_type,
          T& primal_feasibility_lhs,
          T& primal_feasibility_lhs_new,
          T& dual_feasibility_lhs,
          T& dual_feasibility_lhs_new,
          T& new_mu_eq,
          T& new_mu_in,
          T& new_mu_eq_inv,
          T& new_mu_in_inv,
          i64 iter)
{
  bool iteration_condition;
  switch (qpsettings.update_mu_iteration_criteria) {
    case UpdateMuIterationCriteria::FactorizationTime: {
      if (iter == 0) {
        qpwork.timer_between_updates.stop();
        qpwork.timer_between_updates.start();
        qpwork.time_since_last_update_mu =
          qpwork.timer_between_updates.elapsed().user;
      } else {
        qpwork.time_since_last_update_mu =
          qpwork.timer_between_updates.elapsed().user;
      }
      iteration_condition = qpwork.time_since_last_update_mu >
                            qpsettings.percentage_factorization_time_update_mu *
                              qpwork.factorization_time_complete_kkt;
      break;
    }
    case UpdateMuIterationCriteria::FixedNumberIterations: {
      iteration_condition =
        iter - qpwork.last_iteration_update_mu > qpsettings.interval_update_mu;
    }
  }

  if (iteration_condition) {
    T update_ratio_primal_dual = compute_update_ratio_primal_dual(
      qpmodel, qpresults, qpwork, box_constraints, hessian_type);

    bool value_condition =
      update_ratio_primal_dual > qpsettings.threshold_ratio_update_mu ||
      update_ratio_primal_dual < qpsettings.threshold_ratio_update_mu_inv;

    if (value_condition) {
      new_mu_eq = qpresults.info.mu_eq / update_ratio_primal_dual;
      new_mu_in = qpresults.info.mu_in / update_ratio_primal_dual;
      new_mu_eq_inv = qpresults.info.mu_eq_inv * update_ratio_primal_dual;
      new_mu_in_inv = qpresults.info.mu_in_inv * update_ratio_primal_dual;

      new_mu_eq = std::min(std::max(new_mu_eq, qpsettings.mu_min_eq),
                           qpsettings.mu_max_eq);
      new_mu_in = std::min(std::max(new_mu_in, qpsettings.mu_min_in_osqp),
                           qpsettings.mu_max_in);
      new_mu_eq_inv =
        std::min(std::max(new_mu_eq_inv, qpsettings.mu_min_eq_inv),
                 qpsettings.mu_max_eq_inv);
      new_mu_in_inv =
        std::min(std::max(new_mu_in_inv, qpsettings.mu_min_in_inv),
                 qpsettings.mu_max_in_inv_osqp);
    }

    if (primal_feasibility_lhs_new >= primal_feasibility_lhs &&
        dual_feasibility_lhs_new >= dual_feasibility_lhs &&
        (qpresults.info.mu_in <= T(1e-3) || qpresults.info.mu_in >= T(1e5))) {
      new_mu_in = qpsettings.cold_reset_mu_in_osqp;
      new_mu_eq = qpsettings.cold_reset_mu_eq_osqp;
      new_mu_in_inv = qpsettings.cold_reset_mu_in_inv_osqp;
      new_mu_eq_inv = qpsettings.cold_reset_mu_eq_inv_osqp;
    }

    if (qpresults.info.mu_in != new_mu_in ||
        qpresults.info.mu_eq != new_mu_eq) {
      {
        ++qpresults.info.mu_updates;
      }
      mu_update(qpmodel,
                qpresults,
                qpwork,
                n_constraints,
                dense_backend,
                new_mu_eq,
                new_mu_in);
      switch (qpsettings.update_mu_iteration_criteria) {
        case UpdateMuIterationCriteria::FactorizationTime: {
          qpwork.timer_between_updates.stop();
          qpwork.timer_between_updates.start();
          break;
        }
        case UpdateMuIterationCriteria::FixedNumberIterations: {
          qpwork.last_iteration_update_mu = iter;
          break;
        }
      }
      qpresults.info.mu_eq = new_mu_eq;
      qpresults.info.mu_in = new_mu_in;
      qpresults.info.mu_eq_inv = new_mu_eq_inv;
      qpresults.info.mu_in_inv = new_mu_in_inv;
    }
  }
}
/*!
 * Checks the feasibility of the problem at the current step of the ADMM (OSQP)
 * solver.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 */
template<typename T>
bool
is_infeasible(const Settings<T>& qpsettings,
              const Model<T>& qpmodel,
              Results<T>& qpresults,
              Workspace<T>& qpwork,
              const bool box_constraints,
              preconditioner::RuizEquilibration<T>& ruiz,
              const HessianType hessian_type)
{
  // Get the intermediate data from the workspace
  // Computation here as it was done implicitely in the corresponding part
  // of the proxqp version in the function primal_dual_newton_semi_smooth
  Vec<T> dx = qpresults.x - qpwork.x_prev;
  Vec<T> dy = qpresults.y - qpwork.y_prev;
  Vec<T> dz = qpresults.z - qpwork.z_prev;

  auto& Hdx = qpwork.Hdx;
  auto& Adx = qpwork.Adx;
  auto& Cdx = qpwork.Cdx;
  auto& ATdy = qpwork.CTz;

  switch (hessian_type) {
    case HessianType::Zero:
      break;
    case HessianType::Dense:
      Hdx.noalias() =
        qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * dx;
      break;
    case HessianType::Diagonal:
#ifndef NDEBUG
      PROXSUITE_THROW_PRETTY(!qpwork.H_scaled.isDiagonal(),
                             std::invalid_argument,
                             "H is not diagonal.");
#endif
      Hdx.array() = qpwork.H_scaled.diagonal().array() * dx.array();
      break;
  }
  Adx.noalias() = qpwork.A_scaled * dx;
  ATdy.noalias() = qpwork.A_scaled.transpose() * dy;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
  if (qpmodel.n_in > 0) {
    Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
    CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
  }
  if (box_constraints) {
    qpwork.active_part_z.tail(qpmodel.dim) = dz.tail(qpmodel.dim);
    qpwork.active_part_z.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
    CTdz.noalias() += qpwork.active_part_z.tail(qpmodel.dim);

    Cdx.tail(qpmodel.dim) = dx;
    Cdx.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
  }

  // Call to intermadiate functions to check the feasibility
  if (qpresults.info.iter_ext % qpsettings.frequence_infeasibility_check == 0 ||
      qpsettings.primal_infeasibility_solving) {

    bool is_primal_infeasible =
      global_primal_residual_infeasibility(VectorViewMut<T>{ from_eigen, ATdy },
                                           VectorViewMut<T>{ from_eigen, CTdz },
                                           VectorViewMut<T>{ from_eigen, dy },
                                           VectorViewMut<T>{ from_eigen, dz },
                                           qpwork,
                                           qpmodel,
                                           qpsettings,
                                           box_constraints,
                                           ruiz);

    bool is_dual_infeasible =
      global_dual_residual_infeasibility(VectorViewMut<T>{ from_eigen, Adx },
                                         VectorViewMut<T>{ from_eigen, Cdx },
                                         VectorViewMut<T>{ from_eigen, Hdx },
                                         VectorViewMut<T>{ from_eigen, dx },
                                         qpwork,
                                         qpsettings,
                                         qpmodel,
                                         box_constraints,
                                         ruiz);

    if (is_primal_infeasible) {
      qpresults.info.status = QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE;
      return true;
    } else if (is_dual_infeasible) {
      qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
      return true;
    }
  }
  return false;
}
/*!
 * One iteration of the ADMM algorithm adapted in OSQP.
 *
 * Solves the linear system (KKT), then update the primal and dual variables.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
admm_step(const Settings<T>& qpsettings,
          const Model<T>& qpmodel,
          Results<T>& qpresults,
          Workspace<T>& qpwork,
          const bool box_constraints,
          const isize n_constraints,
          const DenseBackend dense_backend)
{
  // Note:
  // In the context of a library (proxsuite) to implement different solvers, we
  // use the same inetrmediate functions (infeasibility, residuals, etc) and API
  // than in the code of ProxQP Yet, the OSQP paper (see
  // https://inria.hal.science/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/)
  // manages both the equality and inequality constraints in one matrix A. Thus,
  // the following adapts the content of the paper to our library (eg matrices
  // A, C, I for constraints).

  // Solve the linear system
  Vec<T> x_tilde;
  Vec<T> nu_eq;
  Vec<T> nu_in;
  Vec<T> zeta_tilde_eq;
  Vec<T> zeta_tilde_in;

  qpwork.rhs.setZero();
  qpwork.rhs.head(qpmodel.dim) =
    qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
    qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;
  qpwork.rhs.tail(n_constraints) =
    qpwork.zeta_in - qpresults.info.mu_in * qpresults.z;

  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + n_constraints;
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  solve_linear_system(qpwork.rhs,
                      qpmodel,
                      qpresults,
                      qpwork,
                      n_constraints,
                      dense_backend,
                      inner_pb_dim,
                      stack);
  x_tilde = qpwork.rhs.head(qpmodel.dim);
  nu_eq = qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq);
  nu_in = qpwork.rhs.tail(n_constraints);

  // Update the variables
  zeta_tilde_eq =
    qpwork.b_scaled + qpresults.info.mu_eq * (nu_eq - qpresults.y);
  zeta_tilde_in = qpwork.zeta_in + qpresults.info.mu_in * (nu_in - qpresults.z);

  qpresults.x =
    qpsettings.alpha_osqp * x_tilde + (1 - qpsettings.alpha_osqp) * qpresults.x;

  qpwork.zeta_eq = qpwork.b_scaled;
  Vec<T> zeta_in_next = qpsettings.alpha_osqp * zeta_tilde_in +
                        (1 - qpsettings.alpha_osqp) * qpwork.zeta_in +
                        qpresults.info.mu_in * qpresults.z;
  if (box_constraints) {
    zeta_in_next.head(qpmodel.n_in) = qpwork.l_scaled.cwiseMax(
      zeta_in_next.head(qpmodel.n_in).cwiseMin(qpwork.u_scaled));
    zeta_in_next.tail(qpmodel.dim) = qpwork.l_box_scaled.cwiseMax(
      zeta_in_next.tail(qpmodel.dim).cwiseMin(qpwork.u_box_scaled));
  } else {
    zeta_in_next =
      qpwork.l_scaled.cwiseMax(zeta_in_next.cwiseMin(qpwork.u_scaled));
  }

  qpresults.y = qpresults.y + qpresults.info.mu_eq_inv * qpsettings.alpha_osqp *
                                (zeta_tilde_eq - qpwork.b_scaled);
  qpresults.z = qpresults.z +
                qpresults.info.mu_in_inv *
                  (qpsettings.alpha_osqp * zeta_tilde_in +
                   (1 - qpsettings.alpha_osqp) * qpwork.zeta_in - zeta_in_next);

  qpwork.zeta_in = zeta_in_next;
}
/*!
 * Iterations of the ADMM algorithm in OSQP.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param iter_start starting index of the first iteration.
 */
template<typename T>
void
admm( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const bool box_constraints,
  isize n_constraints,
  const DenseBackend& dense_backend,
  const HessianType& hessian_type,
  preconditioner::RuizEquilibration<T>& ruiz,
  T& primal_feasibility_eq_rhs_0,
  T& primal_feasibility_in_rhs_0,
  T& primal_feasibility_eq_lhs,
  T& primal_feasibility_in_lhs,
  T& primal_feasibility_lhs,
  T& dual_feasibility_lhs,
  T& dual_feasibility_rhs_0,
  T& dual_feasibility_rhs_1,
  T& dual_feasibility_rhs_3,
  T& rhs_duality_gap,
  T& duality_gap,
  T& scaled_eps,
  T& scaled_eps_rel,
  i64 iter_start)
{
  for (i64 iter = iter_start; iter < qpsettings.max_iter; ++iter) {

    T new_mu_in(qpresults.info.mu_in);
    T new_mu_eq(qpresults.info.mu_eq);
    T new_mu_in_inv(qpresults.info.mu_in_inv);
    T new_mu_eq_inv(qpresults.info.mu_eq_inv);

    bool is_solved_qp =
      proxsuite::common::is_solved(qpsettings,
                                   qpmodel,
                                   qpresults,
                                   qpwork,
                                   box_constraints,
                                   hessian_type,
                                   ruiz,
                                   common::QPSolver::OSQP,
                                   primal_feasibility_eq_rhs_0,
                                   primal_feasibility_in_rhs_0,
                                   primal_feasibility_eq_lhs,
                                   primal_feasibility_in_lhs,
                                   primal_feasibility_lhs,
                                   dual_feasibility_lhs,
                                   dual_feasibility_rhs_0,
                                   dual_feasibility_rhs_1,
                                   dual_feasibility_rhs_3,
                                   rhs_duality_gap,
                                   duality_gap,
                                   scaled_eps,
                                   scaled_eps_rel,
                                   iter);
    if (is_solved_qp) {
      if (iter == 0) {
        qpwork.admm_solved_at_init = true;
      }
      break;
    }

    qpresults.info.iter_ext += 1;

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    proxsuite::common::compute_scaled_primal_residual_ineq(
      qpsettings,
      qpmodel,
      qpresults,
      qpwork,
      box_constraints,
      ruiz,
      common::QPSolver::OSQP);

    admm_step(qpsettings,
              qpmodel,
              qpresults,
              qpwork,
              box_constraints,
              n_constraints,
              dense_backend);

    bool is_infeasible_qp = is_infeasible(qpsettings,
                                          qpmodel,
                                          qpresults,
                                          qpwork,
                                          box_constraints,
                                          ruiz,
                                          hessian_type);
    if (is_infeasible_qp) {
      break;
    }

    qpwork.active_set_up.array() =
      (qpwork.primal_residual_in_scaled_up.array() >
       0); // {zeta_in - u + z > 0}
    qpwork.active_set_low.array() =
      (qpresults.si.array() < 0); // {zeta_in - l + z < 0}

    // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // std::cout << "In admm:" << std::endl;
    // std::cout << "z:" << std::endl << qpresults.z << std::endl;
    // std::cout << "active_set_low:" << std::endl << qpwork.active_set_low <<
    // std::endl; std::cout << "active_set_up:" << std::endl <<
    // qpwork.active_set_up << std::endl; std::cout <<
    // "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    T primal_feasibility_lhs_new(primal_feasibility_lhs);
    T dual_feasibility_lhs_new(dual_feasibility_lhs);
    proxsuite::common::update_solver_status(qpsettings,
                                            qpmodel,
                                            qpresults,
                                            qpwork,
                                            box_constraints,
                                            hessian_type,
                                            ruiz,
                                            primal_feasibility_eq_rhs_0,
                                            primal_feasibility_in_rhs_0,
                                            primal_feasibility_eq_lhs,
                                            primal_feasibility_in_lhs,
                                            primal_feasibility_lhs_new,
                                            dual_feasibility_lhs_new,
                                            dual_feasibility_rhs_0,
                                            dual_feasibility_rhs_1,
                                            dual_feasibility_rhs_3,
                                            rhs_duality_gap,
                                            duality_gap,
                                            scaled_eps,
                                            scaled_eps_rel);

    if (qpsettings.update_mu) {
      update_mu(qpsettings,
                qpmodel,
                qpresults,
                qpwork,
                box_constraints,
                n_constraints,
                dense_backend,
                hessian_type,
                primal_feasibility_lhs,
                primal_feasibility_lhs_new,
                dual_feasibility_lhs,
                dual_feasibility_lhs_new,
                new_mu_eq,
                new_mu_in,
                new_mu_eq_inv,
                new_mu_in_inv,
                iter);
    }

  } // outer iterations loop
}
/*!
 * Find the active sets in solution polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
find_active_sets(const Settings<T>& qpsettings,
                 const Model<T>& qpmodel,
                 Results<T>& qpresults,
                 Workspace<T>& qpwork,
                 const bool box_constraints,
                 preconditioner::RuizEquilibration<T>& ruiz,
                 VecBool& active_constraints_eq,
                 isize& num_active_constraints_eq,
                 isize& num_active_constraints_eq_low,
                 isize& num_active_constraints_eq_up,
                 VecBool& active_constraints_ineq,
                 isize& num_active_constraints_ineq,
                 isize& num_active_constraints_ineq_low,
                 isize& num_active_constraints_ineq_up,
                 isize& num_active_constraints,
                 isize& inner_pb_dim)
{
  // Equality
  qpwork.active_set_low_eq.array() = (qpresults.y.array() < 0);
  qpwork.active_set_up_eq.array() = (qpresults.y.array() > 0);
  active_constraints_eq = qpwork.active_set_up_eq || qpwork.active_set_low_eq;
  num_active_constraints_eq = active_constraints_eq.count();
  num_active_constraints_eq_low = qpwork.active_set_low_eq.count();
  num_active_constraints_eq_up = qpwork.active_set_up_eq.count();

  // Inequality
  if (qpwork.admm_solved_at_init) {
    // Compute active sets that could not be computed in ADMM
    qpwork.z_prev = qpresults.z;
    if (box_constraints) {
      qpwork.zeta_in.head(qpmodel.n_in) = qpwork.C_scaled * qpresults.x;
      qpwork.zeta_in.tail(qpmodel.dim).array() =
        qpwork.i_scaled.array() * qpresults.x.array();
    } else {
      qpwork.zeta_in = qpwork.C_scaled * qpresults.x;
    }

    proxsuite::common::compute_scaled_primal_residual_ineq(
      qpsettings,
      qpmodel,
      qpresults,
      qpwork,
      box_constraints,
      ruiz,
      common::QPSolver::OSQP);

    qpwork.active_set_up.array() =
      (qpwork.primal_residual_in_scaled_up.array() >
       0); // {zeta_in - u + z > 0}
    qpwork.active_set_low.array() =
      (qpresults.si.array() < 0); // {zeta_in - l + z < 0}
  }
  active_constraints_ineq = qpwork.active_set_up || qpwork.active_set_low;
  num_active_constraints_ineq = active_constraints_ineq.count();
  num_active_constraints_ineq_low = qpwork.active_set_low.count();
  num_active_constraints_ineq_up = qpwork.active_set_up.count();

  num_active_constraints =
    num_active_constraints_eq + num_active_constraints_ineq;

  inner_pb_dim = qpmodel.dim + num_active_constraints;

  // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
  // std::cout << "In find_active_sets" << std::endl;
  // std::cout << "y:" << std::endl << qpresults.y << std::endl;
  // std::cout << "active_set_low_eq:" << std::endl << qpwork.active_set_low_eq
  // << std::endl; std::cout << "active_set_up_eq:" << std::endl <<
  // qpwork.active_set_up_eq << std::endl; std::cout << "z:" << std::endl <<
  // qpresults.z << std::endl; std::cout << "active_set_low:" << std::endl <<
  // qpwork.active_set_low << std::endl; std::cout << "active_set_up:" <<
  // std::endl << qpwork.active_set_up << std::endl; std::cout <<
  // "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
}
/*!
 * Build the reduced matrices of constraints in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
build_reduced_constraints_matrices( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const isize n_constraints,
  Mat<T>& A_low,
  Mat<T>& A_up,
  Mat<T>& C_low,
  Mat<T>& C_up)
{
  isize low_index = 0;
  isize up_index = 0;
  for (isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      A_low.row(low_index) = qpwork.A_scaled.row(i);
      ++low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      A_up.row(up_index) = qpwork.A_scaled.row(i);
      ++up_index;
    }
  }

  low_index = 0;
  up_index = 0;
  Vec<T> tmp_low(qpmodel.dim);
  Vec<T> tmp_up(qpmodel.dim);
  tmp_low.setZero();
  tmp_up.setZero();
  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low(i)) {
      if (i < qpmodel.n_in) {
        C_low.row(low_index) = qpwork.C_scaled.row(i);
      } else {
        tmp_low(i - qpmodel.n_in) = qpwork.i_scaled(i - qpmodel.n_in);
        C_low.row(low_index) = tmp_low;
        tmp_low(i - qpmodel.n_in) = 0.;
      }
      ++low_index;
    }
    if (qpwork.active_set_up(i)) {
      if (i < qpmodel.n_in) {
        C_up.row(up_index) = qpwork.C_scaled.row(i);
      } else {
        tmp_up(i - qpmodel.n_in) = qpwork.i_scaled(i - qpmodel.n_in);
        C_up.row(up_index) = tmp_up;
        tmp_up(i - qpmodel.n_in) = 0.;
      }
      ++up_index;
    }
  }
}
/*!
 * Build the matrices K and K + Delta_K in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 */
template<typename T>
void
build_kkt_matrices_polishing( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Workspace<T>& qpwork,
  const HessianType hessian_type,
  Mat<T>& k_polish,
  Mat<T>& k_plus_delta_k_polish,
  Mat<T> A_low,
  Mat<T> A_up,
  Mat<T> C_low,
  Mat<T> C_up,
  isize num_active_constraints_eq_low,
  isize num_active_constraints_ineq_low,
  isize num_active_constraints_eq_up,
  isize num_active_constraints_ineq_up,
  isize num_active_constraints)
{
  // Construction of K
  isize row;
  isize col;

  switch (hessian_type) {
    case HessianType::Dense:
      k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
      break;
    case HessianType::Zero:
      k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim).setZero();
      break;
    case HessianType::Diagonal:
      k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
      break;
  }

  col = qpmodel.dim;
  k_polish.block(0, col, qpmodel.dim, num_active_constraints_eq_low) =
    A_low.transpose();

  col += num_active_constraints_eq_low;
  k_polish.block(0, col, qpmodel.dim, num_active_constraints_ineq_low) =
    C_low.transpose();

  col += num_active_constraints_ineq_low;
  k_polish.block(0, col, qpmodel.dim, num_active_constraints_eq_up) =
    A_up.transpose();

  col += num_active_constraints_eq_up;
  k_polish.block(0, col, qpmodel.dim, num_active_constraints_ineq_up) =
    C_up.transpose();

  row = qpmodel.dim;
  k_polish.block(row, 0, num_active_constraints_eq_low, qpmodel.dim) = A_low;

  row += num_active_constraints_eq_low;
  k_polish.block(row, 0, num_active_constraints_ineq_low, qpmodel.dim) = C_low;

  row += num_active_constraints_ineq_low;
  k_polish.block(row, 0, num_active_constraints_eq_up, qpmodel.dim) = A_up;

  row += num_active_constraints_eq_up;
  k_polish.block(row, 0, num_active_constraints_ineq_up, qpmodel.dim) = C_up;

  k_polish.bottomRightCorner(num_active_constraints, num_active_constraints)
    .setZero();

  // Construction of K + Delta_K
  k_plus_delta_k_polish = k_polish;
  k_plus_delta_k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim)
    .diagonal()
    .array() += qpsettings.delta;
  k_plus_delta_k_polish
    .bottomRightCorner(num_active_constraints, num_active_constraints)
    .diagonal()
    .array() -= qpsettings.delta;
}
/*!
 * Build the right hand side (-g, l_L, u_U) in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 */
template<typename T>
void
build_rhs_polishing( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Workspace<T>& qpwork,
  const HessianType hessian_type,
  const isize n_constraints,
  Vec<T>& rhs_polish,
  isize num_active_constraints_eq_low,
  isize num_active_constraints_ineq_low,
  isize num_active_constraints_eq_up,
  isize num_active_constraints_ineq_up)
{
  isize low_index = 0;
  isize up_index = 0;
  Vec<T> b_low(num_active_constraints_eq_low);
  Vec<T> b_up(num_active_constraints_eq_up);
  for (isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      b_low(low_index) = qpwork.b_scaled(i);
      ++low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      b_up(up_index) = qpwork.b_scaled(i);
      ++up_index;
    }
  }

  low_index = 0;
  up_index = 0;
  Vec<T> l_low(num_active_constraints_ineq_low);
  Vec<T> u_up(num_active_constraints_ineq_up);
  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low(i)) {
      if (i < qpmodel.n_in) {
        l_low(low_index) = qpwork.l_scaled(i);
      } else {
        l_low(low_index) = qpwork.l_box_scaled(i - qpmodel.n_in);
      }
      ++low_index;
    }
    if (qpwork.active_set_up(i)) {
      if (i < qpmodel.n_in) {
        u_up(up_index) = qpwork.u_scaled(i);
      } else {
        u_up(up_index) = qpwork.u_box_scaled(i - qpmodel.n_in);
      }
      ++up_index;
    }
  }

  isize row;
  isize col;

  row = qpmodel.dim;
  rhs_polish.head(row) = -qpwork.g_scaled;
  rhs_polish.segment(row, num_active_constraints_eq_low) = b_low;

  row += num_active_constraints_eq_low;
  rhs_polish.segment(row, num_active_constraints_ineq_low) = l_low;

  row += num_active_constraints_ineq_low;
  rhs_polish.segment(row, num_active_constraints_eq_up) = b_up;
  rhs_polish.tail(num_active_constraints_ineq_up) = u_up;
}
/*!
 * Iterative refinement in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 */
template<typename T>
void
iterative_refinement_polishing( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const DenseBackend& dense_backend,
  const isize n_constraints,
  Vec<T>& hat_t,
  Vec<T> rhs_polish,
  Mat<T> k_polish,
  isize inner_pb_dim,
  plv::dynstack::DynStackMut& stack)
{
  // Solve the reduced system before iterative refinement
  hat_t = rhs_polish;
  solve_linear_system(hat_t,
                      qpmodel,
                      qpresults,
                      qpwork,
                      n_constraints,
                      dense_backend,
                      inner_pb_dim,
                      stack);

  // Iterative refinement
  Vec<T> rhs_polish_refine(inner_pb_dim);
  Vec<T> delta_hat_t(inner_pb_dim);

  for (i64 iter = 0; iter < qpsettings.polish_refine_iter; ++iter) {
    rhs_polish_refine = rhs_polish - k_polish * hat_t;
    delta_hat_t = rhs_polish_refine;

    solve_linear_system(delta_hat_t,
                        qpmodel,
                        qpresults,
                        qpwork,
                        n_constraints,
                        dense_backend,
                        inner_pb_dim,
                        stack);

    hat_t = hat_t + delta_hat_t;
  }
}
/*!
 * Update primal and dual variables in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
void
update_variables_polishing( //
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const bool box_constraints,
  const isize n_constraints,
  Vec<T> hat_t,
  isize num_active_constraints_eq_low,
  isize num_active_constraints_eq_up,
  isize num_active_constraints_ineq_low)
{
  // Get (x, y, z) from hat_t
  qpresults.x = hat_t.head(qpmodel.dim);

  isize low_index = 0;
  isize up_index = 0;
  for (isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      qpresults.y(i) = hat_t(qpmodel.dim + low_index);
      ++low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      qpresults.y(i) = hat_t(qpmodel.dim + num_active_constraints_eq_low +
                             num_active_constraints_ineq_low + up_index);
      ++up_index;
    }
  }

  low_index = 0;
  up_index = 0;
  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low(i)) {
      qpresults.z(i) =
        hat_t(qpmodel.dim + num_active_constraints_eq_low + low_index);
      ++low_index;
    }
    if (qpwork.active_set_up(i)) {
      qpresults.z(i) = hat_t(qpmodel.dim + num_active_constraints_eq_low +
                             num_active_constraints_ineq_low +
                             num_active_constraints_eq_up + up_index);
      ++up_index;
    }
  }

  // Projection of the dual solution into the normal cone N_C(zeta)
  Vec<T> tmp_z(n_constraints);
  tmp_z = qpresults.z + qpwork.zeta_in;
  if (box_constraints) {
    qpwork.zeta_in.head(qpmodel.n_in) = qpwork.l_scaled.cwiseMax(
      qpresults.z.head(qpmodel.n_in).cwiseMin(qpwork.u_scaled));
    qpwork.zeta_in.tail(qpmodel.dim) = qpwork.l_box_scaled.cwiseMax(
      qpresults.z.tail(qpmodel.dim).cwiseMin(qpwork.u_box_scaled));
  } else {
    qpresults.z =
      qpwork.l_scaled.cwiseMax(qpwork.zeta_in.cwiseMin(qpwork.u_scaled));
  }
  qpresults.z = tmp_z - qpwork.zeta_in;
}
/*!
 * Update residuals in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param qpsettings solver settings.
 */
template<typename T>
void
update_residuals_polishing( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const bool box_constraints,
  const isize n_constraints,
  const HessianType hessian_type,
  preconditioner::RuizEquilibration<T>& ruiz,
  T& primal_feasibility_lhs,
  T& primal_feasibility_eq_rhs_0,
  T& primal_feasibility_in_rhs_0,
  T& primal_feasibility_eq_lhs,
  T& primal_feasibility_in_lhs,
  T& dual_feasibility_lhs,
  T& dual_feasibility_rhs_0,
  T& dual_feasibility_rhs_1,
  T& dual_feasibility_rhs_3,
  T& rhs_duality_gap,
  T& duality_gap,
  bool& is_feasible)
{
  global_primal_residual(qpmodel,
                         qpresults,
                         qpsettings,
                         qpwork,
                         ruiz,
                         box_constraints,
                         primal_feasibility_lhs,
                         primal_feasibility_eq_rhs_0,
                         primal_feasibility_in_rhs_0,
                         primal_feasibility_eq_lhs,
                         primal_feasibility_in_lhs);

  bool is_primal_feasible =
    primal_feasibility_lhs <=
    (qpsettings.eps_abs +
     qpsettings.eps_rel *
       std::max(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0));
  qpresults.info.pri_res = primal_feasibility_lhs;

  if (is_primal_feasible) {
    global_dual_residual(qpresults,
                         qpwork,
                         qpmodel,
                         box_constraints,
                         ruiz,
                         dual_feasibility_lhs,
                         dual_feasibility_rhs_0,
                         dual_feasibility_rhs_1,
                         dual_feasibility_rhs_3,
                         rhs_duality_gap,
                         duality_gap,
                         hessian_type);
    qpresults.info.dua_res = dual_feasibility_lhs;
    qpresults.info.duality_gap = duality_gap;

    bool is_dual_feasible =
      dual_feasibility_lhs <=
      (qpsettings.eps_abs +
       qpsettings.eps_rel *
         std::max(
           std::max(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
           std::max(dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2)));

    if (is_dual_feasible) {
      if (qpsettings.check_duality_gap) {
        if (std::fabs(qpresults.info.duality_gap) <=
            qpsettings.eps_duality_gap_abs +
              qpsettings.eps_duality_gap_rel * rhs_duality_gap) {
          is_feasible = true;
        }
      } else {
        is_feasible = true;
      }
    }
  }
}
/*!
 * Check the success of solution polishing.
 *
 * @param qpresults solver results.
 * @param qpsettings solver settings.
 */
template<typename T>
void
check_success_polishing( //
  const Settings<T>& qpsettings,
  Results<T>& qpresults,
  T pri_res_admm,
  T dua_res_admm,
  T duality_gap_admm,
  bool is_feasible,
  bool& polish_success)
{
  if (qpsettings.check_duality_gap) {
    bool polish_success_primal_dual =
      ((qpresults.info.pri_res < pri_res_admm) &&
       (qpresults.info.dua_res < dua_res_admm))

      || ((qpresults.info.pri_res < pri_res_admm) &&
          (qpresults.info.dua_res < 1e-10))

      || ((qpresults.info.dua_res < dua_res_admm) &&
          (qpresults.info.pri_res < 1e-10));

    polish_success =
      ((qpresults.info.duality_gap < duality_gap_admm) &&
       polish_success_primal_dual)

      || ((qpresults.info.duality_gap < 1e-9) && polish_success_primal_dual);
  } else {
    polish_success = ((qpresults.info.pri_res < pri_res_admm) &&
                      (qpresults.info.dua_res < dua_res_admm))

                     || ((qpresults.info.pri_res < pri_res_admm) &&
                         (qpresults.info.dua_res < 1e-10))

                     || ((qpresults.info.dua_res < dua_res_admm) &&
                         (qpresults.info.pri_res < 1e-10));
  }

  if (!is_feasible) {
    polish_success = false;
  }
}
/*!
 * Print polishing line after the ADMM iterations.
 *
 * @param qpresults solver results.
 * @param qpsettings solver settings.
 */
template<typename T>
void
print_polishing_line( //
  const Settings<T>& qpsettings,
  Results<T>& qpresults,
  bool is_feasible)
{
  std::cout << "\033[1;34m[polishing]\033[0m" << std::endl;
  std::cout << std::scientific << std::setw(2) << std::setprecision(2)
            << " | primal residual=" << qpresults.info.pri_res
            << " | dual residual=" << qpresults.info.dua_res
            << " | duality gap=" << qpresults.info.duality_gap
            << " | delta=" << qpsettings.delta << std::endl;
  switch (qpresults.info.polish_status) {
    case PolishStatus::POLISH_SUCCEED: {
      std::cout << "\033[1;34m[polishing: succeed]\033[0m" << std::endl;
      break;
    }
    case PolishStatus::POLISH_FAILED: {
      if (!is_feasible) {
        std::cout << "\033[1;34m[polishing: infeasible solution. "
                  << (qpsettings.resume_admm ? "Resume ADMM" : "") << "]\033[0m"
                  << std::endl;
      } else {
        std::cout << "\033[1;34m[polishing: failed. "
                  << (qpsettings.resume_admm ? "Resume ADMM" : "") << "]\033[0m"
                  << std::endl;
      }
      break;
    }
    case PolishStatus::POLISH_NO_ACTIVE_SET_FOUND: {
      break;
    }
    case PolishStatus::POLISH_NOT_RUN: {
      break;
    }
  }
}
/*!
 * Solution polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
polish(const Settings<T>& qpsettings,
       const Model<T>& qpmodel,
       Results<T>& qpresults,
       Workspace<T>& qpwork,
       const bool box_constraints,
       const isize n_constraints,
       const DenseBackend dense_backend,
       const HessianType hessian_type,
       preconditioner::RuizEquilibration<T>& ruiz,
       T& primal_feasibility_lhs,
       T& primal_feasibility_eq_rhs_0,
       T& primal_feasibility_in_rhs_0,
       T& primal_feasibility_eq_lhs,
       T& primal_feasibility_in_lhs,
       T& dual_feasibility_lhs,
       T& dual_feasibility_rhs_0,
       T& dual_feasibility_rhs_1,
       T& dual_feasibility_rhs_3,
       T& rhs_duality_gap,
       T& duality_gap)
{

  // Timing polishing
  qpwork.timer_polish.stop();
  qpwork.timer_polish.start();

  // ADMM solution
  auto x_admm = qpresults.x;
  auto y_admm = qpresults.y;
  auto z_admm = qpresults.z;
  auto zeta_in_admm = qpwork.zeta_in;

  auto pri_res_admm = qpresults.info.pri_res;
  auto dua_res_admm = qpresults.info.dua_res;
  auto duality_gap_admm = qpresults.info.duality_gap;

  // Upper and lower active constraints
  VecBool active_constraints_eq;
  isize num_active_constraints_eq;
  isize num_active_constraints_eq_low;
  isize num_active_constraints_eq_up;

  VecBool active_constraints_ineq;
  isize num_active_constraints_ineq;
  isize num_active_constraints_ineq_low;
  isize num_active_constraints_ineq_up;

  isize num_active_constraints;
  isize inner_pb_dim;

  find_active_sets(qpsettings,
                   qpmodel,
                   qpresults,
                   qpwork,
                   box_constraints,
                   ruiz,
                   active_constraints_eq,
                   num_active_constraints_eq,
                   num_active_constraints_eq_low,
                   num_active_constraints_eq_up,
                   active_constraints_ineq,
                   num_active_constraints_ineq,
                   num_active_constraints_ineq_low,
                   num_active_constraints_ineq_up,
                   num_active_constraints,
                   inner_pb_dim);

  if (num_active_constraints == 0) {
    qpresults.info.polish_status = PolishStatus::POLISH_NO_ACTIVE_SET_FOUND;
    if (qpsettings.verbose) {
      std::cout << "\033[1;34m[polishing: no active set found. "
                << (qpsettings.resume_admm ? "Resume ADMM" : "") << "]\033[0m"
                << std::endl;
    }
    return;
  }

  // Build the reduced matrices of the constraints
  Mat<T> A_low(num_active_constraints_eq_low, qpmodel.dim);
  Mat<T> A_up(num_active_constraints_eq_up, qpmodel.dim);

  Mat<T> C_low(num_active_constraints_ineq_low, qpmodel.dim);
  Mat<T> C_up(num_active_constraints_ineq_up, qpmodel.dim);

  build_reduced_constraints_matrices(qpsettings,
                                     qpmodel,
                                     qpresults,
                                     qpwork,
                                     n_constraints,
                                     A_low,
                                     A_up,
                                     C_low,
                                     C_up);

  // Build the KKT and regularized KKT matrices
  Mat<T> k_polish(inner_pb_dim, inner_pb_dim);
  Mat<T> k_plus_delta_k_polish(inner_pb_dim, inner_pb_dim);

  build_kkt_matrices_polishing(qpsettings,
                               qpmodel,
                               qpwork,
                               hessian_type,
                               k_polish,
                               k_plus_delta_k_polish,
                               A_low,
                               A_up,
                               C_low,
                               C_up,
                               num_active_constraints_eq_low,
                               num_active_constraints_ineq_low,
                               num_active_constraints_eq_up,
                               num_active_constraints_ineq_up,
                               num_active_constraints);

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    qpwork.ldl_stack.as_mut(),
  };

  qpwork.ldl.factorize(k_plus_delta_k_polish.transpose(), stack);

  // Construction of rhs_polish
  Vec<T> rhs_polish(inner_pb_dim);

  build_rhs_polishing(qpsettings,
                      qpmodel,
                      qpwork,
                      hessian_type,
                      n_constraints,
                      rhs_polish,
                      num_active_constraints_eq_low,
                      num_active_constraints_ineq_low,
                      num_active_constraints_eq_up,
                      num_active_constraints_ineq_up);

  // Iterative refinement
  Vec<T> hat_t(inner_pb_dim);

  iterative_refinement_polishing(qpsettings,
                                 qpmodel,
                                 qpresults,
                                 qpwork,
                                 dense_backend,
                                 n_constraints,
                                 hat_t,
                                 rhs_polish,
                                 k_polish,
                                 inner_pb_dim,
                                 stack);

  // Update of (x, y, z)
  update_variables_polishing(qpmodel,
                             qpresults,
                             qpwork,
                             box_constraints,
                             n_constraints,
                             hat_t,
                             num_active_constraints_eq_low,
                             num_active_constraints_eq_up,
                             num_active_constraints_ineq_low);

  // Timing polishing
  qpwork.time_polishing = qpwork.timer_polish.elapsed().user;

  // Update of residuals
  bool is_feasible = false;

  update_residuals_polishing(qpsettings,
                             qpmodel,
                             qpresults,
                             qpwork,
                             box_constraints,
                             n_constraints,
                             hessian_type,
                             ruiz,
                             primal_feasibility_lhs,
                             primal_feasibility_eq_rhs_0,
                             primal_feasibility_in_rhs_0,
                             primal_feasibility_eq_lhs,
                             primal_feasibility_in_lhs,
                             dual_feasibility_lhs,
                             dual_feasibility_rhs_0,
                             dual_feasibility_rhs_1,
                             dual_feasibility_rhs_3,
                             rhs_duality_gap,
                             duality_gap,
                             is_feasible);

  // Check polish success
  bool polish_success;

  check_success_polishing(qpsettings,
                          qpresults,
                          pri_res_admm,
                          dua_res_admm,
                          duality_gap_admm,
                          is_feasible,
                          polish_success);

  if (polish_success) {
    qpresults.info.polish_status = PolishStatus::POLISH_SUCCEED;
  } else {
    qpresults.info.polish_status = PolishStatus::POLISH_FAILED;
  }

  // Print polishing line
  if (qpsettings.verbose) {
    print_polishing_line(qpsettings, qpresults, is_feasible);
  }

  // Go back if polish failed
  if (!polish_success) {
    qpresults.x = x_admm;
    qpresults.y = y_admm;
    qpresults.z = z_admm;
    qpwork.zeta_in = zeta_in_admm;

    qpresults.info.pri_res = pri_res_admm;
    qpresults.info.dua_res = dua_res_admm;
    qpresults.info.duality_gap = duality_gap_admm;
  }
}
/*!
 * Executes the OSQP algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 */
template<typename T>
void
qp_solve( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const bool box_constraints,
  const DenseBackend& dense_backend,
  const HessianType& hessian_type,
  preconditioner::RuizEquilibration<T>& ruiz)
{
  PROXSUITE_EIGEN_MALLOC_NOT_ALLOWED();

  // Setup
  proxsuite::common::setup_solver(qpsettings,
                                  qpmodel,
                                  qpresults,
                                  qpwork,
                                  box_constraints,
                                  dense_backend,
                                  hessian_type,
                                  ruiz,
                                  common::QPSolver::OSQP);

  isize n_constraints(qpmodel.n_in);
  if (box_constraints) {
    n_constraints += qpmodel.dim;
  }

  T primal_feasibility_eq_rhs_0(0);
  T primal_feasibility_in_rhs_0(0);
  T dual_feasibility_rhs_0(0);
  T dual_feasibility_rhs_1(0);
  T dual_feasibility_rhs_3(0);
  T primal_feasibility_lhs(0);
  T primal_feasibility_eq_lhs(0);
  T primal_feasibility_in_lhs(0);
  T dual_feasibility_lhs(0);

  T duality_gap(0);
  T rhs_duality_gap(0);

  bool polish_solution = qpsettings.polish;
  if (qpmodel.n_in + qpmodel.n_eq == 0) {
    polish_solution = false;
  }

  T scaled_eps;
  T scaled_eps_rel;
  if (polish_solution) {
    if (qpsettings.high_accuracy) {
      scaled_eps = 1e-5;
      scaled_eps_rel = (qpsettings.eps_rel == 0) ? 0 : 1e-5;
    } else {
      scaled_eps = 1e-3;
      scaled_eps_rel = (qpsettings.eps_rel == 0) ? 0 : 1e-3;
    }
  } else {
    scaled_eps = qpsettings.eps_abs;
    scaled_eps_rel = qpsettings.eps_rel;
  }

  // Body of solve
  admm(qpsettings,
       qpmodel,
       qpresults,
       qpwork,
       box_constraints,
       n_constraints,
       dense_backend,
       hessian_type,
       ruiz,
       primal_feasibility_eq_rhs_0,
       primal_feasibility_in_rhs_0,
       primal_feasibility_eq_lhs,
       primal_feasibility_in_lhs,
       primal_feasibility_lhs,
       dual_feasibility_lhs,
       dual_feasibility_rhs_0,
       dual_feasibility_rhs_1,
       dual_feasibility_rhs_3,
       rhs_duality_gap,
       duality_gap,
       scaled_eps,
       scaled_eps_rel,
       0);

  if (polish_solution) {
    if (qpresults.info.status == QPSolverOutput::PROXQP_SOLVED) {
      polish(qpsettings,
             qpmodel,
             qpresults,
             qpwork,
             box_constraints,
             n_constraints,
             dense_backend,
             hessian_type,
             ruiz,
             primal_feasibility_lhs,
             primal_feasibility_eq_rhs_0,
             primal_feasibility_in_rhs_0,
             primal_feasibility_eq_lhs,
             primal_feasibility_in_lhs,
             dual_feasibility_lhs,
             dual_feasibility_rhs_0,
             dual_feasibility_rhs_1,
             dual_feasibility_rhs_3,
             rhs_duality_gap,
             duality_gap);

      bool resume =
        qpsettings.resume_admm &&
        (qpresults.info.polish_status ==
           PolishStatus::POLISH_NO_ACTIVE_SET_FOUND ||
         qpresults.info.polish_status == PolishStatus::POLISH_FAILED);

      if (resume) {
        if (qpresults.info.polish_status == PolishStatus::POLISH_FAILED) {
          proxsuite::proxqp::dense::setup_factorization(
            qpwork, qpmodel, qpresults, dense_backend, hessian_type);
          proxsuite::common::setup_factorization_complete_kkt(
            qpwork, qpmodel, qpresults, dense_backend, n_constraints);
        }
        scaled_eps = qpsettings.eps_abs;
        scaled_eps_rel = qpsettings.eps_rel;
        admm(qpsettings,
             qpmodel,
             qpresults,
             qpwork,
             box_constraints,
             n_constraints,
             dense_backend,
             hessian_type,
             ruiz,
             primal_feasibility_eq_rhs_0,
             primal_feasibility_in_rhs_0,
             primal_feasibility_eq_lhs,
             primal_feasibility_in_lhs,
             primal_feasibility_lhs,
             dual_feasibility_lhs,
             dual_feasibility_rhs_0,
             dual_feasibility_rhs_1,
             dual_feasibility_rhs_3,
             rhs_duality_gap,
             duality_gap,
             scaled_eps,
             scaled_eps_rel,
             qpresults.info.iter_ext);
      }
    }
  }

  // End of solve
  proxsuite::common::unscale_solver(
    qpsettings, qpmodel, qpresults, box_constraints, ruiz);
  proxsuite::common::compute_objective(qpmodel, qpresults);
  if (qpsettings.compute_timings) {
    proxsuite::common::compute_timings(qpresults, qpwork);
  }

  if (qpsettings.verbose) {
    proxsuite::common::print_solver_statistics(
      qpsettings, qpresults, common::QPSolver::OSQP);
  }

  proxsuite::common::prepare_next_solve(qpresults, qpwork);

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */