//
// Copyright (c) 2022 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_SOLVER_HPP
#define PROXSUITE_OSQP_DENSE_SOLVER_HPP

#include "proxsuite/linalg/veg/internal/typedefs.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/dense/preconditioner/ruiz.hpp"
#include "proxsuite/solvers/common/utils.hpp"
#include <proxsuite/linalg/dense/ldlt.hpp>
#include "proxsuite/proxqp/dense/solver.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
#include <iostream>

namespace proxsuite {
namespace osqp {
namespace dense {

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::dense;
using namespace proxsuite::solvers::utils;

/*!
 * Derives the global primal residual of the QP problem.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param scaled_primal_feasibility_lhs scaled primal infeasibility.
 * @param scaled_primal_feasibility_eq_rhs_0 scaled scalar variable used when
 * using a relative stopping criterion.
 * @param scaled_primal_feasibility_in_rhs_0 scaled scalar variable used when
 * using a relative stopping criterion.
 * @param scaled_primal_feasibility_eq_lhs scaled scalar variable used when
 * using a relative stopping criterion.
 * @param scaled_primal_feasibility_in_lhs scaled scalar variable used when
 * using a relative stopping criterion.
 */
template<typename T>
void
scaled_global_primal_residual(const Model<T>& qpmodel,
                              Results<T>& qpresults,
                              Workspace<T>& qpwork,
                              const preconditioner::RuizEquilibration<T>& ruiz,
                              const bool box_constraints,
                              T& scaled_primal_feasibility_lhs,
                              T& scaled_primal_feasibility_eq_rhs_0,
                              T& scaled_primal_feasibility_in_rhs_0,
                              T& scaled_primal_feasibility_eq_lhs,
                              T& scaled_primal_feasibility_in_lhs)
{

  // Function inspired from global_primal_residual, but:
  // We compute the scaled terms
  // We add the norm of zeta (z in osqp paper) in the max for the primal
  // residual criteria

  qpresults.se.noalias() = qpwork.A_scaled * qpresults.x;
  scaled_primal_feasibility_eq_rhs_0 = infty_norm(qpresults.se);

  qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in).noalias() =
    qpwork.C_scaled * qpresults.x;
  if (box_constraints) {
    qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) = qpresults.x;
  }
  scaled_primal_feasibility_in_rhs_0 =
    infty_norm(qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in));

  ruiz.unscale_primal_residual_in_place_in(VectorViewMut<T>{
    from_eigen, qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in) });
  if (box_constraints) {
    ruiz.unscale_primal_in_place(VectorViewMut<T>{
      from_eigen, qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) });
  }

  qpresults.si.head(qpmodel.n_in) =
    helpers::positive_part(
      qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in) - qpmodel.u) +
    helpers::negative_part(
      qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in) - qpmodel.l);
  ruiz.scale_primal_residual_in_place_in(
    VectorViewMut<T>{ from_eigen, qpresults.si.head(qpmodel.n_in) });
  if (box_constraints) {
    qpresults.si.tail(qpmodel.dim) =
      helpers::positive_part(
        qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) - qpmodel.u_box) +
      helpers::negative_part(
        qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) - qpmodel.l_box);
    ruiz.scale_primal_residual_in_place(
      VectorViewMut<T>{ from_eigen, qpresults.si.tail(qpmodel.dim) });
    qpwork.active_part_z.tail(qpmodel.dim) =
      qpresults.x - qpresults.si.tail(qpmodel.dim);
    scaled_primal_feasibility_in_rhs_0 =
      std::max(scaled_primal_feasibility_in_rhs_0,
               infty_norm(qpwork.active_part_z.tail(qpmodel.dim)));
    scaled_primal_feasibility_in_rhs_0 =
      std::max(scaled_primal_feasibility_in_rhs_0, infty_norm(qpresults.x));
  }

  ruiz.unscale_primal_residual_in_place_eq(
    VectorViewMut<T>{ from_eigen, qpresults.se });
  qpresults.se -= qpmodel.b;
  ruiz.scale_primal_residual_in_place_eq(
    VectorViewMut<T>{ from_eigen, qpresults.se });
  scaled_primal_feasibility_eq_lhs = infty_norm(qpresults.se);

  scaled_primal_feasibility_in_lhs = infty_norm(qpresults.si);
  ruiz.unscale_primal_residual_in_place_in(
    VectorViewMut<T>{ from_eigen, qpresults.si.head(qpmodel.n_in) });
  if (box_constraints) {
    ruiz.unscale_primal_residual_in_place(
      VectorViewMut<T>{ from_eigen, qpresults.si.tail(qpmodel.dim) });
  }

  scaled_primal_feasibility_lhs = std::max(scaled_primal_feasibility_eq_lhs,
                                           scaled_primal_feasibility_in_lhs);
}
/*!
 * Derives the global dual residual of the QP problem.
 *
 * @param qpwork solver workspace.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param dual_feasibility_lhs primal infeasibility.
 * @param primal_feasibility_eq_rhs_0 scaled scalar variable used when using a
 * relative stopping criterion.
 * @param dual_feasibility_rhs_0 scaled scalar variable used when using a
 * relative stopping criterion.
 * @param dual_feasibility_rhs_1 scaled scalar variable used when using a
 * relative stopping criterion.
 * @param dual_feasibility_rhs_3 scaled scalar variable used when using a
 * relative stopping criterion.
 */
template<typename T>
void
scaled_global_dual_residual(Results<T>& qpresults,
                            Workspace<T>& qpwork,
                            const Model<T>& qpmodel,
                            const bool box_constraints,
                            T& scaled_dual_feasibility_lhs,
                            T& scaled_dual_feasibility_rhs_0,
                            T& scaled_dual_feasibility_rhs_1,
                            T& scaled_dual_feasibility_rhs_3,
                            const HessianType& hessian_type)
{

  qpwork.dual_residual_scaled = qpwork.g_scaled;

  switch (hessian_type) {
    case HessianType::Zero:
      scaled_dual_feasibility_rhs_0 = 0;
      break;
    case HessianType::Dense:
      qpwork.CTz.noalias() =
        qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * qpresults.x;
      qpwork.dual_residual_scaled += qpwork.CTz;
      scaled_dual_feasibility_rhs_0 = infty_norm(qpwork.CTz);
      break;
    case HessianType::Diagonal:
      qpwork.CTz.array() =
        qpwork.H_scaled.diagonal().array() * qpresults.x.array();
      qpwork.dual_residual_scaled += qpwork.CTz;
      scaled_dual_feasibility_rhs_0 = infty_norm(qpwork.CTz);
      break;
  }

  qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * qpresults.y;
  qpwork.dual_residual_scaled += qpwork.CTz;
  scaled_dual_feasibility_rhs_1 = infty_norm(qpwork.CTz);

  qpwork.CTz.noalias() =
    qpwork.C_scaled.transpose() * qpresults.z.head(qpmodel.n_in);
  qpwork.dual_residual_scaled += qpwork.CTz;
  scaled_dual_feasibility_rhs_3 = infty_norm(qpwork.CTz);
  if (box_constraints) {
    qpwork.CTz.noalias() = qpresults.z.tail(qpmodel.dim);
    qpwork.CTz.array() *= qpwork.i_scaled.array();

    qpwork.dual_residual_scaled += qpwork.CTz;
    scaled_dual_feasibility_rhs_3 =
      std::max(infty_norm(qpwork.CTz), scaled_dual_feasibility_rhs_3);
  }

  scaled_dual_feasibility_lhs = infty_norm(qpwork.dual_residual_scaled);
}
/*!
 * Boolean to check if we update the proximal parameters mu_eq and mu_in.
 *
 * @param qpwork workspace of the solver.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solution results.
 */
template<typename T>
T
compute_update_mu_criteria(Workspace<T>& qpwork,
                           const Model<T>& qpmodel,
                           Results<T>& qpresults,
                           const HessianType& hessian_type,
                           const preconditioner::RuizEquilibration<T>& ruiz,
                           const bool box_constraints,
                           T& scaled_primal_feasibility_lhs,
                           T& scaled_primal_feasibility_eq_rhs_0,
                           T& scaled_primal_feasibility_in_rhs_0,
                           T& scaled_primal_feasibility_eq_lhs,
                           T& scaled_primal_feasibility_in_lhs,
                           T& scaled_dual_feasibility_lhs,
                           T& scaled_dual_feasibility_rhs_0,
                           T& scaled_dual_feasibility_rhs_1,
                           T& scaled_dual_feasibility_rhs_3)
{

  scaled_global_primal_residual(qpmodel,
                                qpresults,
                                qpwork,
                                ruiz,
                                box_constraints,
                                scaled_primal_feasibility_lhs,
                                scaled_primal_feasibility_eq_rhs_0,
                                scaled_primal_feasibility_in_rhs_0,
                                scaled_primal_feasibility_eq_lhs,
                                scaled_primal_feasibility_in_lhs);

  scaled_global_dual_residual(qpresults,
                              qpwork,
                              qpmodel,
                              box_constraints,
                              scaled_dual_feasibility_lhs,
                              scaled_dual_feasibility_rhs_0,
                              scaled_dual_feasibility_rhs_1,
                              scaled_dual_feasibility_rhs_3,
                              hessian_type);

  T epsilon =
    1e-9; // TODO: Better way to code this (to avoid dividing by zero) ?

  T primal_term =
    (scaled_primal_feasibility_lhs + epsilon) /
    (std::max(scaled_primal_feasibility_eq_rhs_0,
              std::max(infty_norm(qpwork.zeta_eq),
                       std::max(infty_norm(qpwork.zeta_in),
                                scaled_primal_feasibility_in_rhs_0))) +
     epsilon);

  T dual_term = (scaled_dual_feasibility_lhs + epsilon) /
                (std::max(scaled_dual_feasibility_rhs_0,
                          std::max(scaled_dual_feasibility_rhs_1,
                                   std::max(infty_norm(qpwork.g_scaled),
                                            scaled_dual_feasibility_rhs_3))) +
                 epsilon);

  T update_mu_criteria = std::sqrt(primal_term / dual_term);

  return update_mu_criteria;
}
/*!
 * Setups and performs the factorization of the full regularized KKT matrix of
 * the problem (containing equality and inequality constraints).
 * It adds n_constraints (n_in + dim if box_constraints) rows/columns to the
 * equality case KKT matrix.
 *
 * @param qpwork workspace of the solver.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solution results.
 */
template<typename T>
void
setup_factorization_full_kkt(Workspace<T>& qpwork,
                             const Model<T>& qpmodel,
                             Results<T>& qpresults,
                             const DenseBackend dense_backend,
                             const isize n_constraints)
{

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  {
    auto _planned_to_add = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, n_constraints);

    T mu_in_neg(-qpresults.info.mu_in);

    {
      switch (dense_backend) {
        case DenseBackend::PrimalDualLDLT: {
          isize n = qpmodel.dim;
          isize n_eq = qpmodel.n_eq;
          LDLT_TEMP_MAT_UNINIT(
            T, new_cols, n + n_eq + n_constraints, n_constraints, stack);
          for (isize k = 0; k < n_constraints; ++k) {
            auto col = new_cols.col(k);
            if (k >= qpmodel.n_in) {
              col.head(n).setZero();
              col[k - qpmodel.n_in] = qpwork.i_scaled[k - qpmodel.n_in];
            } else {
              col.head(n) = (qpwork.C_scaled.row(k));
            }
            col.tail(n_eq + n_constraints).setZero();
            col[n + n_eq + k] = mu_in_neg;
          }
          qpwork.timer_full_fact.start();
          qpwork.ldl.insert_block_at(n + n_eq, new_cols, stack);
          qpwork.timer_full_fact.stop();
          qpwork.factorization_time_full_kkt =
            qpwork.timer_full_fact.elapsed().user;
        } break;
        case DenseBackend::PrimalLDLT: {
          LDLT_TEMP_MAT_UNINIT(T, new_cols, qpmodel.dim, n_constraints, stack);
          qpwork.dw_aug.head(n_constraints).setOnes();
          qpwork.dw_aug.head(n_constraints).array() *= qpresults.info.mu_in_inv;
          for (isize i = 0; i < n_constraints; ++i) {
            auto col = new_cols.col(i);
            if (i >= qpmodel.n_in) {
              col.setZero();
              col[i - qpmodel.n_in] = qpwork.i_scaled[i - qpmodel.n_in];
            } else {
              col.head(qpmodel.dim) = qpwork.C_scaled.row(i);
            }
          }
          qpwork.ldl.rank_r_update(
            new_cols, qpwork.dw_aug.head(n_constraints), stack);
        } break;
        case DenseBackend::Automatic: {
        } break;
      }
    }
  }

  qpwork.n_c = n_constraints;
}
/*!
 * Checks the feasibility of the problem at the current step of the solver
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
check_infeasibility(const Settings<T>& qpsettings,
                    const Model<T>& qpmodel,
                    Results<T>& qpresults,
                    Workspace<T>& qpwork,
                    const bool box_constraints,
                    preconditioner::RuizEquilibration<T>& ruiz,
                    const HessianType hessian_type)
{
  Vec<T> dx = qpresults.x - qpwork.x_prev;
  Vec<T> dy = qpresults.y - qpwork.y_prev;
  Vec<T> dz = qpresults.z - qpwork.z_prev;

  auto& Hdx = qpwork.Hdx;
  auto& Adx = qpwork.Adx;
  auto& ATdy = qpwork.CTz;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

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
  ATdy.noalias() = qpwork.A_scaled.transpose() * dy;
  Adx.noalias() = qpwork.A_scaled * dx;

  auto& Cdx = qpwork.Cdx;
  LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
  if (qpmodel.n_in > 0) {
    Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
    CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
  }
  // Add box constraints
  if (box_constraints) {

    qpwork.active_part_z.tail(qpmodel.dim) = dz.tail(qpmodel.dim);
    qpwork.active_part_z.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
    CTdz.noalias() += qpwork.active_part_z.tail(qpmodel.dim);

    Cdx.tail(qpmodel.dim) = dx;
    Cdx.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
  }

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
  } else if (is_dual_infeasible) {
    qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
  }

  if ((qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE &&
       !qpsettings.primal_infeasibility_solving) ||
      qpresults.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
    return true;
  } else {
    return false;
  }
}
/*!
 * Iteration of ADMM step.
 * Solves the KKT system with equality and inequality constraints, then update
 * the primal and dual variables
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
admm(const Settings<T>& qpsettings,
     const Model<T>& qpmodel,
     Results<T>& qpresults,
     Workspace<T>& qpwork,
     const bool box_constraints,
     const isize n_constraints,
     preconditioner::RuizEquilibration<T>& ruiz,
     const DenseBackend dense_backend,
     const HessianType hessian_type)
{

  // Note:
  // It is convenient, given the architecture of proxsuite, to reuse the same
  // functions (infeasibility, residuals, etc) and wrapper than the proxqp
  // solver. Given the different problem formulation (see
  // https://inria.hal.science/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/),
  // the variables y and z in osqp are adapted to our framework, namely with
  // zeta_eq = z_osqp[eq:], zeta_in = z_osqp[:in] and y_osqp = concat(y_proxqp,
  // z_proxqp).

  // Corpus of the ADMM iteration
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  Vec<T> x_tilde;
  Vec<T> nu_eq;
  Vec<T> nu_in;
  Vec<T> zeta_tilde_eq;
  Vec<T> zeta_tilde_in;

  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT: {
      // Construction of the rhs
      qpwork.rhs.setZero();
      qpwork.rhs.head(qpmodel.dim) =
        qpresults.info.rho * qpresults.x - qpwork.g_scaled;
      qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
        qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;
      qpwork.rhs.tail(n_constraints) =
        qpwork.zeta_in - qpresults.info.mu_in * qpresults.z;

      // Solve the linear system
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
                          SolverType::OSQP,
                          inner_pb_dim,
                          stack);

      // Updates of the intermediate variables for x, y and z
      x_tilde = qpwork.rhs.head(qpmodel.dim);

      nu_eq = qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq);
      nu_in = qpwork.rhs.tail(n_constraints);
      zeta_tilde_eq =
        qpwork.zeta_eq + qpresults.info.mu_eq * (nu_eq - qpresults.y);
      zeta_tilde_in =
        qpwork.zeta_in + qpresults.info.mu_in * (nu_in - qpresults.z);

    } break;
    case DenseBackend::PrimalLDLT: {
      // Construction of the rhs
      qpwork.rhs.setZero();
      qpwork.rhs.resize(qpmodel.dim);
      qpwork.rhs = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
      qpwork.rhs += qpwork.A_scaled.transpose() *
                    (qpresults.info.mu_eq_inv * qpwork.b_scaled - qpresults.y);
      qpwork.rhs +=
        qpwork.C_scaled.transpose() *
        (qpresults.info.mu_in_inv * qpwork.zeta_in.head(qpmodel.n_in) -
         qpresults.z.head(qpmodel.n_in));
      if (box_constraints) {
        qpwork.rhs += qpwork.i_scaled.cwiseProduct(
          qpresults.info.mu_in_inv *
          (qpwork.zeta_in.tail(qpmodel.dim) - qpresults.z.tail(qpmodel.dim)));
      }

      // Solve the linear system
      isize inner_pb_dim = qpmodel.dim;
      proxsuite::linalg::veg::dynstack::DynStackMut stack{
        proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
      };
      solve_linear_system(qpwork.rhs,
                          qpmodel,
                          qpresults,
                          qpwork,
                          n_constraints,
                          dense_backend,
                          SolverType::OSQP,
                          inner_pb_dim,
                          stack);

      // Updates of the intermediate variables for x, y and z
      x_tilde = qpwork.rhs;

      zeta_tilde_eq = qpwork.A_scaled * x_tilde;
      zeta_tilde_in.resize(n_constraints);
      zeta_tilde_in.head(qpmodel.n_in) = qpwork.C_scaled * x_tilde;
      if (box_constraints) {
        zeta_tilde_in.tail(qpmodel.dim) = qpwork.i_scaled.cwiseProduct(x_tilde);
      }

    } break;
  }

  // Update of x
  qpresults.x =
    qpsettings.alpha_osqp * x_tilde + (1 - qpsettings.alpha_osqp) * qpresults.x;

  // Update of the zeta's (orthogonal projection on the constraints)
  Vec<T> new_zeta_eq = qpwork.b_scaled;
  Vec<T> sum_zeta_in_z = qpsettings.alpha_osqp * zeta_tilde_in +
                         (1 - qpsettings.alpha_osqp) * qpwork.zeta_in +
                         qpresults.info.mu_in * qpresults.z;
  Vec<T> new_zeta_in;
  new_zeta_in.resize(n_constraints);
  if (box_constraints) {
    new_zeta_in.head(qpmodel.n_in) = qpwork.l_scaled.cwiseMax(
      sum_zeta_in_z.head(qpmodel.n_in).cwiseMin(qpwork.u_scaled));
    new_zeta_in.tail(qpmodel.dim) = qpwork.l_box_scaled.cwiseMax(
      sum_zeta_in_z.tail(qpmodel.dim).cwiseMin(qpwork.u_box_scaled));
  } else {
    new_zeta_in =
      qpwork.l_scaled.cwiseMax(sum_zeta_in_z.cwiseMin(qpwork.u_scaled));
  }

  // Update of y and z
  qpresults.y = qpresults.y +
                qpresults.info.mu_eq_inv *
                  (qpsettings.alpha_osqp * zeta_tilde_eq +
                   (1 - qpsettings.alpha_osqp) * qpwork.zeta_eq - new_zeta_eq);
  qpresults.z = qpresults.z +
                qpresults.info.mu_in_inv *
                  (qpsettings.alpha_osqp * zeta_tilde_in +
                   (1 - qpsettings.alpha_osqp) * qpwork.zeta_in - new_zeta_in);

  // Update of the zeta's
  qpwork.zeta_eq = new_zeta_eq;
  qpwork.zeta_in = new_zeta_in;
}
/*!
 * Polishing step after the ADMM.
 * Iterative refinement of the solution from the system K + Delta_K * hat_t =
 * rhs.
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
       const isize n_constraints,
       const DenseBackend dense_backend,
       const HessianType hessian_type)
{

  std::cout << "check 01" << std::endl;

  // Find the upper and lower active constraints
  qpwork.active_set_up_eq.array() = (qpresults.y.array() > 0);
  qpwork.active_set_low_eq.array() = (qpresults.y.array() < 0);
  qpwork.active_constraints_eq =
    qpwork.active_set_up_eq || qpwork.active_set_low_eq;
  isize numactive_constraints_eq = qpwork.active_constraints_eq.count();
  isize numactive_constraints_eq_up = qpwork.active_set_up_eq.count();
  isize numactive_constraints_eq_low = qpwork.active_set_low_eq.count();

  std::cout << "check 02" << std::endl;

  qpwork.active_set_up_ineq = qpwork.active_set_up;
  qpwork.active_set_low_ineq = qpwork.active_set_low;
  qpwork.active_constraints_ineq =
    qpwork.active_set_up_ineq || qpwork.active_set_low_ineq;
  isize numactive_constraints_ineq = qpwork.active_constraints_ineq.count();
  isize numactive_constraints_ineq_up = qpwork.active_set_up_ineq.count();
  isize numactive_constraints_ineq_low = qpwork.active_set_low_ineq.count();

  std::cout << "check 03" << std::endl;

  isize numactive_constraints =
    numactive_constraints_eq + numactive_constraints_ineq;

  // Construction of the matrices A and C of lower and upper active constraints
  Mat<T> C_low(numactive_constraints_ineq_low, qpmodel.dim);
  Mat<T> C_up(numactive_constraints_ineq_up, qpmodel.dim);

  isize low_index = 0;
  isize up_index = 0;
  Vec<T> tmp_low;
  Vec<T> tmp_up;
  tmp_low.setZero(qpmodel.dim);
  tmp_up.setZero(qpmodel.dim);
  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low_ineq(i)) {
      if (i < qpmodel.n_in) {
        C_low.row(low_index) = qpwork.C_scaled.row(i);
      } else {
        tmp_low(i - qpmodel.n_in) = qpwork.i_scaled(i - qpmodel.n_in);
        C_low.row(low_index) = tmp_low;
        tmp_low(i - qpmodel.n_in) = 0;
      }
      ++low_index;
    }
    if (qpwork.active_set_up_ineq(i)) {
      if (i < qpmodel.n_in) {
        C_up.row(up_index) = qpwork.C_scaled.row(i);
      } else {
        tmp_up(i - qpmodel.n_in) = qpwork.i_scaled(i - qpmodel.n_in);
        C_up.row(up_index) = tmp_up;
        tmp_up(i - qpmodel.n_in) = 0;
      }
      ++up_index;
    }
  }

  std::cout << "check 04" << std::endl;

  Mat<T> A_low(numactive_constraints_eq_low, qpmodel.dim);
  Mat<T> A_up(numactive_constraints_eq_up, qpmodel.dim);

  low_index = 0;
  up_index = 0;
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

  std::cout << "check 05" << std::endl;

  // Construction and factorization of K + Delta_K
  isize row;
  isize column;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

  std::cout << "check 06" << std::endl;

  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT:

      // Top left corner of K for H
      qpwork.k_polish.resize(qpmodel.dim + numactive_constraints,
                             qpmodel.dim + numactive_constraints);

      std::cout << "check 07" << std::endl;

      switch (hessian_type) {
        case HessianType::Dense:
          qpwork.k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim) =
            qpwork.H_scaled;
          break;
        case HessianType::Zero:
          qpwork.k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim).setZero();
          break;
        case HessianType::Diagonal:
          qpwork.k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim) =
            qpwork.H_scaled;
          break;
      }

      std::cout << "check 08" << std::endl;

      // Add the rows/columns for the constraints
      row = qpmodel.dim;
      std::cout << "check 081" << std::endl;
      column = qpmodel.dim;
      std::cout << "check 082" << std::endl;
      qpwork.k_polish.block(
        0, column, qpmodel.dim, numactive_constraints_eq_low) =
        A_low.transpose();
      std::cout << "check 083" << std::endl;
      column += numactive_constraints_eq_low;
      qpwork.k_polish.block(
        0, column, qpmodel.dim, numactive_constraints_ineq_low) =
        C_low.transpose();
      std::cout << "check 084" << std::endl;
      column += numactive_constraints_ineq_low;
      qpwork.k_polish.block(
        0, column, qpmodel.dim, numactive_constraints_eq_up) = A_up.transpose();
      column += numactive_constraints_eq_up;
      std::cout << "check 085" << std::endl;
      qpwork.k_polish.block(
        0, column, qpmodel.dim, numactive_constraints_ineq_up) =
        C_up.transpose();
      qpwork.k_polish.block(row, 0, numactive_constraints_eq_low, qpmodel.dim) =
        A_low;
      std::cout << "check 086" << std::endl;
      row += numactive_constraints_eq_low;
      qpwork.k_polish.block(
        row, 0, numactive_constraints_ineq_low, qpmodel.dim) = C_low;
      row += numactive_constraints_ineq_low;
      std::cout << "check 087" << std::endl;
      qpwork.k_polish.block(row, 0, numactive_constraints_eq_up, qpmodel.dim) =
        A_up;
      row += numactive_constraints_eq_up;
      // std::cout << "check 088" << std::endl;
      qpwork.k_polish.block(
        row, 0, numactive_constraints_ineq_up, qpmodel.dim) = C_up;
      // std::cout << "check 089" << std::endl;
      qpwork.k_polish
        .bottomRightCorner(numactive_constraints, numactive_constraints)
        .setZero();

      std::cout << "check 09" << std::endl;

      // std::cout << "check 1" << std::endl;
      // std::cout << "checkout 1" << std::endl;
      // Construction and factorization of K + Delta_K
      qpwork.k_plus_delta_k_polish =
        qpwork.k_polish; // error: malloc(): invalid size (unsorted), malloc():
                         // unaligned tcache chunk detected
      // errors here in test cvxpy, but also maros meszaros for some problems
      // std::cout << "check 2" << std::endl;
      // std::cout << "checkout 2" << std::endl;
      std::cout << "check 091" << std::endl;
      qpwork.k_plus_delta_k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim)
        .diagonal()
        .array() += qpsettings.delta_osqp;
      std::cout << "check 092" << std::endl;
      qpwork.k_plus_delta_k_polish
        .bottomRightCorner(numactive_constraints, numactive_constraints)
        .diagonal()
        .array() -= qpsettings.delta_osqp;

      std::cout << "check 010" << std::endl;

      qpwork.ldl.factorize(qpwork.k_plus_delta_k_polish.transpose(), stack);

      break;
    case DenseBackend::PrimalLDLT:

      // Top left corner of K for H -> Initialize to H
      qpwork.k_plus_delta_k_polish.resize(qpmodel.dim, qpmodel.dim);
      switch (hessian_type) {
        case HessianType::Dense:
          qpwork.k_plus_delta_k_polish = qpwork.H_scaled;
          break;
        case HessianType::Zero:
          qpwork.k_plus_delta_k_polish.setZero();
          break;
        case HessianType::Diagonal:
          qpwork.k_plus_delta_k_polish = qpwork.H_scaled;
          break;
      }

      // Construction and factorization of K + Delta_K -> Add the delta terms
      qpwork.k_plus_delta_k_polish.diagonal().array() += qpsettings.delta_osqp;
      qpwork.k_plus_delta_k_polish +=
        qpsettings.delta_osqp_inv * A_low.transpose() * A_low +
        qpsettings.delta_osqp_inv * C_low.transpose() * C_low +
        qpsettings.delta_osqp_inv * A_up.transpose() * A_up +
        qpsettings.delta_osqp_inv * C_up.transpose() * C_up;

      qpwork.ldl.factorize(qpwork.k_plus_delta_k_polish.transpose(), stack);

      break;
  }

  std::cout << "check 011" << std::endl;

  // Construction of the l_low, u_up, etc to build the rhs for the linear
  // systems (iterative refinement)
  isize l_low_index = 0;
  isize u_up_index = 0;
  Vec<T> l_low(numactive_constraints_ineq_low);
  Vec<T> u_up(numactive_constraints_ineq_up);
  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low_ineq(i)) {
      if (i < qpmodel.n_in) {
        l_low(l_low_index) = qpwork.l_scaled(i);
      } else {
        l_low(l_low_index) = qpwork.l_box_scaled(i - qpmodel.n_in);
      }
      ++l_low_index;
    }
    if (qpwork.active_set_up_ineq(i)) {
      if (i < qpmodel.n_in) {
        u_up(u_up_index) = qpwork.u_scaled(i);
      } else {
        u_up(u_up_index) = qpwork.u_box_scaled(i - qpmodel.n_in);
      }
      ++u_up_index;
    }
  }

  std::cout << "check 012" << std::endl;

  isize b_low_index = 0;
  isize b_up_index = 0;
  Vec<T> b_low(numactive_constraints_eq_low);
  Vec<T> b_up(numactive_constraints_eq_up);
  for (isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      b_low(b_low_index) = qpwork.b_scaled(i);
      ++b_low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      b_up(b_up_index) = qpwork.b_scaled(i);
      ++b_up_index;
    }
  }

  std::cout << "check 013" << std::endl;

  // Iterative refinement
  Vec<T> g_polish_rhs;
  isize line;
  isize inner_pb_dim = qpmodel.dim + numactive_constraints;
  Vec<T> hat_t;
  Vec<T> hat_x;
  Vec<T> delta_hat_x;
  Vec<T> rhs_polish;
  Vec<T> rhs_polish_1;
  Vec<T> rhs_polish_2;
  Vec<T> delta_hat_t;
  isize y_low_index;
  isize z_low_index;
  isize y_up_index;
  isize z_up_index;
  Vec<T> hat_y_low;
  Vec<T> hat_z_low;
  Vec<T> hat_y_up;
  Vec<T> hat_z_up;

  std::cout << "check 014" << std::endl;

  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT:

      // First linear system before the iterative refinement to get the first
      // hat_t
      g_polish_rhs.resize(qpmodel.dim + numactive_constraints);
      line = qpmodel.dim;
      g_polish_rhs.head(line) = -qpwork.g_scaled;
      g_polish_rhs.segment(line, numactive_constraints_eq_low) = b_low;
      line += numactive_constraints_eq_low;
      g_polish_rhs.segment(line, numactive_constraints_ineq_low) = l_low;
      line += numactive_constraints_ineq_low;
      g_polish_rhs.segment(line, numactive_constraints_eq_up) = b_up;
      g_polish_rhs.tail(numactive_constraints_ineq_up) = u_up;

      inner_pb_dim = qpmodel.dim + numactive_constraints;

      // std::cout << "check 3" << std::endl;
      std::cout << "check 015" << std::endl;
      hat_t = g_polish_rhs; // 2 times Fatal glibc error: malloc.c:4376
                            // (_int_malloc): assertion failed: (unsigned long)
                            // (size) >= (unsigned long) (nb)
      // std::cout << "check 4" << std::endl;
      std::cout << "check 016" << std::endl;

      solve_linear_system(hat_t,
                          qpmodel,
                          qpresults,
                          qpwork,
                          n_constraints,
                          dense_backend,
                          SolverType::OSQP,
                          inner_pb_dim,
                          stack);

      std::cout << "check 017" << std::endl;

      // Iterative refinement
      for (i64 iter = 0; iter < qpsettings.nb_polish_iter; ++iter) {
        rhs_polish = g_polish_rhs - qpwork.k_polish * hat_t;
        delta_hat_t = rhs_polish;
        solve_linear_system(delta_hat_t,
                            qpmodel,
                            qpresults,
                            qpwork,
                            n_constraints,
                            dense_backend,
                            SolverType::OSQP,
                            inner_pb_dim,
                            stack);
        hat_t = hat_t + delta_hat_t;
      }

      std::cout << "check 018" << std::endl;

      break;
    case DenseBackend::PrimalLDLT:

      // First solve before the iterative refinement (to initialize the rhs of
      // the latter)
      g_polish_rhs.resize(qpmodel.dim);
      g_polish_rhs = -qpwork.g_scaled;
      g_polish_rhs += qpsettings.delta_osqp_inv * A_low.transpose() * b_low +
                      qpsettings.delta_osqp_inv * C_low.transpose() * l_low +
                      qpsettings.delta_osqp_inv * A_up.transpose() * b_up +
                      qpsettings.delta_osqp_inv * C_up.transpose() * u_up;

      inner_pb_dim = qpmodel.dim;
      hat_x = g_polish_rhs;

      solve_linear_system(hat_x,
                          qpmodel,
                          qpresults,
                          qpwork,
                          n_constraints,
                          dense_backend,
                          SolverType::OSQP,
                          inner_pb_dim,
                          stack);

      hat_y_low.resize(
        numactive_constraints_eq_low); // TODO: See if it is mandatory to put
                                       // this resize, or if it is done
                                       // implicitly
      hat_y_low = qpsettings.delta_osqp_inv * (A_low * hat_x - b_low);
      hat_z_low.resize(numactive_constraints_ineq_low);
      hat_z_low = qpsettings.delta_osqp_inv * (C_low * hat_x - l_low);
      hat_y_up.resize(numactive_constraints_eq_up);
      hat_y_up = qpsettings.delta_osqp_inv * (A_up * hat_x - b_up);
      hat_z_up.resize(numactive_constraints_ineq_up);
      hat_z_up = qpsettings.delta_osqp_inv * (C_up * hat_x - u_up);

      // Iterative refinement
      for (i64 iter = 0; iter < qpsettings.nb_polish_iter; ++iter) {
        rhs_polish = g_polish_rhs;
        rhs_polish_1 =
          (qpwork.H_scaled * hat_x + A_low.transpose() * hat_y_low +
           C_low.transpose() * hat_z_low + A_up.transpose() * hat_y_up +
           C_up.transpose() * hat_z_up);
        rhs_polish_2 = qpsettings.delta_osqp_inv *
                       (A_low.transpose() * A_low + C_low.transpose() * C_low +
                        A_up.transpose() * A_up + C_up.transpose() * C_up) *
                       hat_x;
        rhs_polish -= rhs_polish_1;
        rhs_polish -= rhs_polish_2;
        delta_hat_x = rhs_polish;

        solve_linear_system(delta_hat_x,
                            qpmodel,
                            qpresults,
                            qpwork,
                            n_constraints,
                            dense_backend,
                            SolverType::OSQP,
                            inner_pb_dim,
                            stack);

        hat_x += delta_hat_x;

        hat_y_low += qpsettings.delta_osqp_inv * (A_low * hat_x - b_low);
        hat_z_low += qpsettings.delta_osqp_inv * (C_low * hat_x - l_low);
        hat_y_up += qpsettings.delta_osqp_inv * (A_up * hat_x - b_up);
        hat_z_up += qpsettings.delta_osqp_inv * (C_up * hat_x - u_up);
      }

      // Construct the hat_t vector to update the primal and dual variables
      hat_t.resize(qpmodel.dim + numactive_constraints);
      line = qpmodel.dim;
      hat_t.head(line) = hat_x;
      hat_t.segment(line, numactive_constraints_eq_low) = hat_y_low;
      line += numactive_constraints_eq_low;
      hat_t.segment(line, numactive_constraints_ineq_low) = hat_z_low;
      line += numactive_constraints_ineq_low;
      hat_t.segment(line, numactive_constraints_eq_up) = hat_y_up;
      hat_t.tail(numactive_constraints_ineq_up) = hat_z_up;

      break;
  }

  // Update of the primal and dual variables
  qpresults.x = hat_t.head(qpmodel.dim);
  std::cout << "check 019" << std::endl;

  y_low_index = 0;
  z_low_index = 0;
  y_up_index = 0;
  z_up_index = 0;
  for (isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      qpresults.y(i) = hat_t(qpmodel.dim + y_low_index);
      ++y_low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      qpresults.y(i) = hat_t(qpmodel.dim + numactive_constraints_eq_low +
                             numactive_constraints_ineq_low + y_up_index);
      ++y_up_index;
    }
  }

  std::cout << "check 020" << std::endl;

  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low_ineq(i)) {
      qpresults.z(i) =
        hat_t(qpmodel.dim + numactive_constraints_eq_low + z_low_index);
      ++z_low_index;
    }
    if (qpwork.active_set_up_ineq(i)) {
      qpresults.z(i) = hat_t(qpmodel.dim + numactive_constraints_eq_low +
                             numactive_constraints_ineq_low +
                             numactive_constraints_eq_up + z_up_index);
      ++z_up_index;
    }
  }

  std::cout << "check 021" << std::endl;
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

  // Initialization

  setup_solver(qpsettings,
               qpmodel,
               qpresults,
               qpwork,
               box_constraints,
               dense_backend,
               hessian_type,
               ruiz,
               SolverType::OSQP);

  // Solve

  isize n_constraints(qpmodel.n_in);
  if (box_constraints) {
    n_constraints += qpmodel.dim;
  }

  T primal_feasibility_eq_rhs_0(0); // norm(unscaled(Ax))
  T primal_feasibility_in_rhs_0(0); // norm(unscaled(Cx))
  T dual_feasibility_rhs_0(0);      // norm(unscaled(Hx))
  T dual_feasibility_rhs_1(0);      // norm(unscaled(ATy))
  T dual_feasibility_rhs_3(0);      // norm(unscaled(CTz))
  T primal_feasibility_eq_lhs(0);   // norm(unscaled(Ax - b))
  T primal_feasibility_in_lhs(0);   // norm(unscaled([Cx - u]+ + [Cx - l]-))
  T primal_feasibility_lhs(
    0); // max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs)
  T dual_feasibility_lhs(0); // norm(scaled(Hx + g + ATy + CTz))

  T duality_gap(0);                 // xHx + gTx + bTy + zTu + zTl
  T rhs_duality_gap(0);             // max(abs(each term))
  T scaled_eps(qpsettings.eps_abs); // eps_abs

  // Scaled variables used for the updates of mu_eq and mu_in
  T scaled_primal_feasibility_lhs(0);
  T scaled_primal_feasibility_eq_rhs_0(0);
  T scaled_primal_feasibility_in_rhs_0(0);
  T scaled_primal_feasibility_eq_lhs(0);
  T scaled_primal_feasibility_in_lhs(0);

  T scaled_dual_feasibility_lhs(0);
  T scaled_dual_feasibility_rhs_0(0);
  T scaled_dual_feasibility_rhs_1(0);
  T scaled_dual_feasibility_rhs_3(0);

  setup_factorization_full_kkt(
    qpwork, qpmodel, qpresults, dense_backend, n_constraints);

  for (i64 iter = 0; iter < qpsettings.max_iter; ++iter) {

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // General

    bool stopping_criteria_before_iter =
      compute_residuals_and_infeasibility_before_iter(
        qpsettings,
        qpmodel,
        qpresults,
        qpwork,
        box_constraints,
        hessian_type,
        ruiz,
        SolverType::OSQP,
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
        iter);

    if (stopping_criteria_before_iter) {
      break;
    }

    qpresults.info.iter_ext += 1; // We start a new external loop update

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // General: Construction of the active sets of constraints (for dual gap in
    // the loop)

    compute_primal_residual_in_scaled_up(qpsettings,
                                         qpmodel,
                                         qpresults,
                                         qpwork,
                                         box_constraints,
                                         ruiz,
                                         SolverType::OSQP);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Specific: Corpus of the solver

    admm(qpsettings,
         qpmodel,
         qpresults,
         qpwork,
         box_constraints,
         n_constraints,
         ruiz,
         dense_backend,
         hessian_type);

    bool stopping_criteria_after_iter = check_infeasibility(qpsettings,
                                                            qpmodel,
                                                            qpresults,
                                                            qpwork,
                                                            box_constraints,
                                                            ruiz,
                                                            hessian_type);

    if (stopping_criteria_after_iter) {
      break;
    }

    if (scaled_eps == qpsettings.eps_abs &&
        qpsettings.primal_infeasibility_solving &&
        qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
      qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq + qpmodel.n_in)
        .setConstant(T(1));
      qpwork.rhs.head(qpmodel.dim).noalias() =
        qpmodel.A.transpose() * qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) +
        qpmodel.C.transpose() *
          qpwork.rhs.segment(qpmodel.dim + qpmodel.n_eq, qpmodel.n_in);
      if (box_constraints) {
        qpwork.rhs.head(qpmodel.dim).array() += qpwork.i_scaled.array();
      }
      scaled_eps =
        infty_norm(qpwork.rhs.head(qpmodel.dim)) * qpsettings.eps_abs;
    } // TODO Copy-Paste: This was copy pasted from the proxqp solver -> clean
      // code
      // TODO: Check if it works once the primal infeasibility issue is fixed

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // General

    T primal_feasibility_lhs_new(primal_feasibility_lhs);

    compute_residuals_and_infeasibility_after_iter(qpsettings,
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
                                                   dual_feasibility_lhs,
                                                   dual_feasibility_rhs_0,
                                                   dual_feasibility_rhs_1,
                                                   dual_feasibility_rhs_3,
                                                   rhs_duality_gap,
                                                   duality_gap,
                                                   scaled_eps);

    if (qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
      std::cout << "Primal infeasible" << std::endl;
    } else if (qpresults.info.status ==
               QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE) {
      std::cout << "Solved closest primal feasible" << std::endl;
    } else if (qpresults.info.status == QPSolverOutput::PROXQP_SOLVED) {
      std::cout << "Solved" << std::endl;
    }
    // TODO: This is a copy paste -> common function

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Specific: Update the proximal parameters mu_eq and mu_in

    if (qpsettings.update_mu_osqp) {

      if (iter == 0) {
        qpwork.timer_mu_update.stop();
        qpwork.timer_mu_update.start();
        qpwork.time_since_last_mu_update = 0;
      }

      qpwork.time_since_last_mu_update = qpwork.timer_mu_update.elapsed().user;
      bool time_condition =
        qpwork.time_since_last_mu_update >
        qpsettings.ratio_time_mu_update * qpwork.factorization_time_full_kkt;

      if (time_condition) {

        T update_mu_criteria =
          compute_update_mu_criteria(qpwork,
                                     qpmodel,
                                     qpresults,
                                     hessian_type,
                                     ruiz,
                                     box_constraints,
                                     scaled_primal_feasibility_lhs,
                                     scaled_primal_feasibility_eq_rhs_0,
                                     scaled_primal_feasibility_in_rhs_0,
                                     scaled_primal_feasibility_eq_lhs,
                                     scaled_primal_feasibility_in_lhs,
                                     scaled_dual_feasibility_lhs,
                                     scaled_dual_feasibility_rhs_0,
                                     scaled_dual_feasibility_rhs_1,
                                     scaled_dual_feasibility_rhs_3);

        bool update_mu =
          update_mu_criteria < qpsettings.ratio_value_mu_update_inv ||
          update_mu_criteria > qpsettings.ratio_value_mu_update;
        if (update_mu) {

          // Compute the new values of mu_eq and mu_in
          T new_mu_eq = qpresults.info.mu_eq / update_mu_criteria;
          T new_mu_in = qpresults.info.mu_in / update_mu_criteria;
          T new_mu_eq_inv = qpresults.info.mu_eq_inv * update_mu_criteria;
          T new_mu_in_inv = qpresults.info.mu_in_inv * update_mu_criteria;

          // Update the parameters in the KKT matrix
          ++qpresults.info.mu_updates;
          mu_update(qpmodel,
                    qpresults,
                    qpwork,
                    n_constraints,
                    dense_backend,
                    SolverType::OSQP,
                    new_mu_eq,
                    new_mu_in);

          qpresults.info.mu_eq = new_mu_eq;
          qpresults.info.mu_in = new_mu_in;
          qpresults.info.mu_eq_inv = new_mu_eq_inv;
          qpresults.info.mu_in_inv = new_mu_in_inv;
        }
      }
    }
  }

  // Polishing
  if (qpsettings.polish) {
    if (qpresults.info.status == QPSolverOutput::PROXQP_SOLVED) {

      polish(qpsettings,
             qpmodel,
             qpresults,
             qpwork,
             n_constraints,
             dense_backend,
             hessian_type);

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

      qpresults.info.pri_res = primal_feasibility_lhs;
      qpresults.info.dua_res = dual_feasibility_lhs;
      qpresults.info.duality_gap = duality_gap;
    }
  }

  // End

  end_qp_solve_and_prepare_next(qpsettings,
                                qpmodel,
                                qpresults,
                                qpwork,
                                box_constraints,
                                ruiz,
                                SolverType::OSQP);

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense

} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */

// TODO: Finish entirely the dense backend (fix all the TODOs)

// TODO: About specific cases
// - proxqp with primal_infeasibility_solving = true -> returns Solved instead
// of Solved the closest ...
// - osqp with same settings: Goes to the maximum number of iterations (idea:
// osqp with just admm is less
//   accurate that proxqp by construction, and the polishing allows to improve
//   the accuracy. One idea is to do the polish step once we get a primal
//   infeasible, then check if passed according to the new
//   primal_feasibility_lhs and scaled_eps)
// - mu_update: test in osqp_dense_qp_eq shows that using the update option is
// better than not using it
//   in 45 cases over 50, and the remaining cases are really close in terms of
//   performance -> see if it is expected that the performances are better in
//   100% of the cases
// - one test fails with the mu_update option, and passes without (in
// osqp_dense_qp_solve, test on warm starts)

// TODO: Lists of tasks to clean the code:
// - copy-pasted code: Put in common or use code from one function only
// - later: put the functions in proxqp used in osqp too (like
// solve_linear_system) in common files
// - appropriate description commentaries on top of the functions (order and all
// parameters)
// - I use the same settings files (structure), but some settings in OSQP should
// not be in PROXQP, and vice versa -> clean it
