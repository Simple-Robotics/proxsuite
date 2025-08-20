//
// Copyright (c) 2025 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_SOLVER_HPP
#define PROXSUITE_OSQP_DENSE_SOLVER_HPP

#include "proxsuite/fwd.hpp"
#include "proxsuite/osqp/dense/aliases.hpp"
#include "proxsuite/common/dense/helpers.hpp"
#include "proxsuite/common/dense/utils.hpp"
#include "proxsuite/common/dense/prints.hpp"

#include <iostream>
#include <iomanip>

namespace proxsuite {
namespace osqp {
namespace dense {

/*!
 * One iteration of the ADMM algorithm adapted in OSQP.
 *
 * Solves the linear system (KKT), then update the primal and dual variables.
 *
 * @param qpsettings solver settings.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param qpwork solver workspace.
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
  // Solve the linear system
  qpwork.x_tilde.setZero();
  qpwork.nu_eq.setZero();
  qpwork.nu_in.setZero();
  qpwork.zeta_tilde_eq.setZero();
  qpwork.zeta_tilde_in.setZero();
  qpwork.zeta_in_next.setZero();

  qpwork.rhs.setZero();
  qpwork.rhs.head(qpmodel.dim) =
    qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
    qpresults.zeta_eq - qpresults.info.mu_eq * qpresults.y;
  qpwork.rhs.tail(n_constraints) =
    qpresults.zeta_in - qpresults.info.mu_in * qpresults.z;

  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + n_constraints;
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  common::dense::solve_linear_system(qpwork.rhs,
                                     qpmodel,
                                     qpresults,
                                     qpwork,
                                     n_constraints,
                                     dense_backend,
                                     inner_pb_dim,
                                     stack);
  qpwork.x_tilde = qpwork.rhs.head(qpmodel.dim);
  qpwork.nu_eq = qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq);
  qpwork.nu_in = qpwork.rhs.tail(n_constraints);

  // Update the variables
  qpwork.zeta_tilde_eq =
    qpresults.zeta_eq + qpresults.info.mu_eq * (qpwork.nu_eq - qpresults.y);
  qpwork.zeta_tilde_in =
    qpresults.zeta_in + qpresults.info.mu_in * (qpwork.nu_in - qpresults.z);

  qpresults.x =
    qpsettings.alpha * qpwork.x_tilde + (1 - qpsettings.alpha) * qpresults.x;

  qpwork.zeta_eq_next = qpwork.b_scaled; // projection in [b, b]
  qpwork.zeta_in_next = qpsettings.alpha * qpwork.zeta_tilde_in +
                        (1 - qpsettings.alpha) * qpresults.zeta_in +
                        qpresults.info.mu_in * qpresults.z;
  if (box_constraints) {
    qpwork.zeta_in_next.head(qpmodel.n_in) = qpwork.l_scaled.cwiseMax(
      qpwork.zeta_in_next.head(qpmodel.n_in).cwiseMin(qpwork.u_scaled));
    qpwork.zeta_in_next.tail(qpmodel.dim) = qpwork.l_box_scaled.cwiseMax(
      qpwork.zeta_in_next.tail(qpmodel.dim).cwiseMin(qpwork.u_box_scaled));
  } else {
    qpwork.zeta_in_next =
      qpwork.l_scaled.cwiseMax(qpwork.zeta_in_next.cwiseMin(qpwork.u_scaled));
  }

  qpresults.y = qpresults.y + qpresults.info.mu_eq_inv *
                                (qpsettings.alpha * qpwork.zeta_tilde_eq +
                                 (1 - qpsettings.alpha) * qpresults.zeta_eq -
                                 qpwork.zeta_eq_next);
  qpresults.z = qpresults.z + qpresults.info.mu_in_inv *
                                (qpsettings.alpha * qpwork.zeta_tilde_in +
                                 (1 - qpsettings.alpha) * qpresults.zeta_in -
                                 qpwork.zeta_in_next);

  qpresults.zeta_eq = qpwork.zeta_eq_next;
  qpresults.zeta_in = qpwork.zeta_in_next;
}

/*!
 * Derives the scaled global primal residual of the QP problem.
 * Computed as OSQP source code does to compute the ratio for mu update.
 *
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param qpwork solver workspace.
 * @param scaled_primal_feasibility_lhs primal infeasibility.
 * @param scaled_primal_feasibility_eq_rhs_0 norm(scaled Ax)
 * @param scaled_primal_feasibility_in_rhs_0 norm(scaled Cx)
 * @param scaled_primal_feasibility_eq_lhs norm(scaled Ax - zeta_eq)
 * @param scaled_primal_feasibility_in_lhs norm(scaled Cx - zeta_in)
 */
template<typename T>
void
scaled_global_primal_residual(const Model<T>& qpmodel,
                              Results<T>& qpresults,
                              Workspace<T>& qpwork,
                              const bool box_constraints,
                              T& scaled_primal_feasibility_lhs,
                              T& scaled_primal_feasibility_eq_rhs_0,
                              T& scaled_primal_feasibility_in_rhs_0,
                              T& scaled_primal_feasibility_eq_lhs,
                              T& scaled_primal_feasibility_in_lhs)
{
  qpresults.se.noalias() = qpwork.A_scaled * qpresults.x;
  qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in).noalias() =
    qpwork.C_scaled * qpresults.x;
  if (box_constraints) {
    qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) = qpresults.x;
  }

  scaled_primal_feasibility_eq_rhs_0 = infty_norm(qpresults.se);
  scaled_primal_feasibility_in_rhs_0 =
    infty_norm(qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in));

  qpresults.si.head(qpmodel.n_in) =
    qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in) -
    qpresults.zeta_in.head(qpmodel.n_in);
  if (box_constraints) {
    qpresults.si.tail(qpmodel.dim) =
      qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) -
      qpresults.zeta_in.tail(qpmodel.dim);

    qpwork.active_part_z.tail(qpmodel.dim) =
      qpresults.x - qpresults.si.tail(qpmodel.dim);
    scaled_primal_feasibility_in_rhs_0 =
      std::max(scaled_primal_feasibility_in_rhs_0,
               infty_norm(qpwork.active_part_z.tail(qpmodel.dim)));
    scaled_primal_feasibility_in_rhs_0 =
      std::max(scaled_primal_feasibility_in_rhs_0, infty_norm(qpresults.x));
  }
  qpresults.se -= qpwork.b_scaled;

  scaled_primal_feasibility_in_lhs = infty_norm(qpresults.si);
  scaled_primal_feasibility_eq_lhs = infty_norm(qpresults.se);
  scaled_primal_feasibility_lhs = std::max(scaled_primal_feasibility_eq_lhs,
                                           scaled_primal_feasibility_in_lhs);
}

/*!
 * Derives the scaled global dual residual of the QP problem.
 * Computed as OSQP source code does to compute the ratio for mu update.
 *
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpwork solver workspace.
 * @param qpresults solver results.
 * @param dual_feasibility_lhs primal infeasibility.
 * @param scaled_dual_feasibility_eq_rhs_0 scalar variable used when using a
 * relative stopping criterion.
 * @param scaled_dual_feasibility_rhs_0 scalar variable used when using a
 * relative stopping criterion.
 * @param scaled_dual_feasibility_rhs_1 scalar variable used when using a
 * relative stopping criterion.
 * @param scaled_dual_feasibility_rhs_3 scalar variable used when using a
 * relative stopping criterion.
 */
template<typename T>
void
scaled_global_dual_residual(
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const bool box_constraints,
  T& scaled_dual_feasibility_lhs,   // norm(scaled dual residual)
  T& scaled_dual_feasibility_rhs_0, // norm(Hx)
  T& scaled_dual_feasibility_rhs_1, // norm(ATy)
  T& scaled_dual_feasibility_rhs_3, // norm(CTz)
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
 * Finds the active sets of constraints for the polishing step.
 *
 * Equality constraints are considred as lower inequality constraints.
 *
 * @param qpsettings solver settings.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param qpwork solver workspace.
 */
template<typename T>
void
find_active_sets(const Settings<T>& qpsettings,
                 const Model<T>& qpmodel,
                 Results<T>& qpresults,
                 Workspace<T>& qpwork,
                 const bool box_constraints,
                 isize& numactive_inequalities,
                 isize& numactive_lower_inequalities,
                 isize& numactive_upper_inequalities,
                 isize& inner_pb_dim)
{
  qpwork.primal_residual_in_scaled_up = qpresults.zeta_in + qpwork.z_prev;
  qpresults.si = qpwork.primal_residual_in_scaled_up;
  qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in) -= qpwork.u_scaled;
  qpresults.si.head(qpmodel.n_in) -= qpwork.l_scaled;
  if (box_constraints) {
    qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) -=
      qpwork.u_box_scaled;
    qpresults.si.tail(qpmodel.dim) -= qpwork.l_box_scaled;
  }

  qpwork.active_set_up.array() =
    (qpwork.primal_residual_in_scaled_up.array() > 0);
  qpwork.active_set_low.array() = (qpresults.si.array() < 0);

  qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;
  numactive_lower_inequalities = qpwork.active_set_low.count();
  numactive_upper_inequalities = qpwork.active_set_up.count();
  numactive_inequalities = qpwork.active_inequalities.count();
  inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
}

/*!
 * Build the reduced matrix of inequality constraints in polishing.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
build_reduced_inequality_constraints_matrices( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const isize n_constraints,
  Mat<T>& C_low,
  Mat<T>& C_up)
{
  isize low_index = 0;
  isize up_index = 0;

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
  Mat<T> C_low,
  Mat<T> C_up,
  isize numactive_lower_inequalities,
  isize numactive_upper_inequalities,
  isize numactive_inequalities)
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
  k_polish.block(0, col, qpmodel.dim, qpmodel.n_eq) =
    qpwork.A_scaled.transpose();

  col += qpmodel.n_eq;
  k_polish.block(0, col, qpmodel.dim, numactive_lower_inequalities) =
    C_low.transpose();

  col += numactive_lower_inequalities;
  k_polish.block(0, col, qpmodel.dim, numactive_upper_inequalities) =
    C_up.transpose();

  row = qpmodel.dim;
  k_polish.block(row, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;

  row += qpmodel.n_eq;
  k_polish.block(row, 0, numactive_lower_inequalities, qpmodel.dim) = C_low;

  row += numactive_lower_inequalities;
  k_polish.block(row, 0, numactive_upper_inequalities, qpmodel.dim) = C_up;

  k_polish
    .bottomRightCorner(qpmodel.n_eq + numactive_inequalities,
                       qpmodel.n_eq + numactive_inequalities)
    .setZero();

  // Construction of K + Delta_K
  k_plus_delta_k_polish = k_polish;
  k_plus_delta_k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim)
    .diagonal()
    .array() += qpsettings.delta;
  k_plus_delta_k_polish
    .bottomRightCorner(qpmodel.n_eq + numactive_inequalities,
                       qpmodel.n_eq + numactive_inequalities)
    .diagonal()
    .array() -= qpsettings.delta;
}

/*!
 * Build the right hand side (-g, b, l_L, u_U) in polishing.
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
  isize numactive_lower_inequalities,
  isize numactive_upper_inequalities)
{
  isize low_index = 0;
  isize up_index = 0;

  Vec<T> l_low(numactive_lower_inequalities);
  Vec<T> u_up(numactive_upper_inequalities);
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

  row = qpmodel.dim;
  rhs_polish.head(row) = -qpwork.g_scaled;
  rhs_polish.segment(row, qpmodel.n_eq) = qpwork.b_scaled;

  row += qpmodel.n_eq;
  rhs_polish.segment(row, numactive_lower_inequalities) = l_low;

  row += numactive_lower_inequalities;
  rhs_polish.segment(row, numactive_upper_inequalities) = u_up;
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
  isize numactive_lower_inequalities)
{
  // Get (x, y, z) from hat_t
  qpresults.x = hat_t.head(qpmodel.dim);
  qpresults.y = hat_t.segment(qpmodel.dim, qpmodel.n_eq);

  isize low_index = 0;
  isize up_index = 0;

  for (isize i = 0; i < n_constraints; ++i) {
    if (qpwork.active_set_low(i)) {
      qpresults.z(i) = hat_t(qpmodel.dim + qpmodel.n_eq + low_index);
      ++low_index;
    }
    if (qpwork.active_set_up(i)) {
      qpresults.z(i) = hat_t(qpmodel.dim + qpmodel.n_eq +
                             numactive_lower_inequalities + up_index);
      ++up_index;
    }
  }

  // Projection of the dual solution into the normal cone N_[l, u](zeta)
  // by doing: z <- z + zeta;  zeta <- proj_[l, u](z);  z <- z - zeta
  qpresults.z += qpresults.zeta_in;
  if (box_constraints) {
    qpresults.zeta_in.head(qpmodel.n_in) = qpwork.l_scaled.cwiseMax(
      qpresults.z.head(qpmodel.n_in).cwiseMin(qpwork.u_scaled));
    qpresults.zeta_in.tail(qpmodel.dim) = qpwork.l_box_scaled.cwiseMax(
      qpresults.z.tail(qpmodel.dim).cwiseMin(qpwork.u_box_scaled));
  } else {
    qpresults.zeta_in =
      qpwork.l_scaled.cwiseMax(qpresults.z.cwiseMin(qpwork.u_scaled));
  }
  qpresults.z -= qpresults.zeta_in;
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
  Results<T>& qpresults)
{
  switch (qpresults.info.status_polish) {
    case PolishStatus::POLISH_SUCCEEDED: {
      std::cout << "\033[1;34m[polishing]\033[0m" << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | delta=" << qpsettings.delta << std::endl;
      std::cout << "\033[1;34m[polishing: succeed]\033[0m" << std::endl;
      break;
    }
    case PolishStatus::POLISH_FAILED: {
      std::cout << "\033[1;34m[polishing]\033[0m" << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | delta=" << qpsettings.delta << std::endl;
      std::cout << "\033[1;34m[polishing: failed]\033[0m" << std::endl;
      break;
    }
    case PolishStatus::POLISH_NO_ACTIVE_SET_FOUND: {
      std::cout << "\033[1;34m[polishing: no active set found]\033[0m"
                << std::endl;
      break;
    }
    case PolishStatus::POLISH_NOT_RUN: {
      std::cout << "\033[1;34m[polishing: not run]\033[0m" << std::endl;
      break;
    }
  }
}

/*!
 * Executes the OSQP algorithm.
 *
 * @param qpsettings solver settings.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param qpwork solver workspace.
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
  common::dense::preconditioner::RuizEquilibration<T>& ruiz)
{
  PROXSUITE_EIGEN_MALLOC_NOT_ALLOWED();

  isize n_constraints(qpmodel.n_in);
  if (box_constraints) {
    n_constraints += qpmodel.dim;
  }
  if (qpsettings.compute_timings) {
    qpwork.timer.stop();
    qpwork.timer.start();
  }

  // Setup header
  ///////////////////////

  if (qpsettings.verbose) {
    common::dense::print_setup_header(qpsettings,
                                      qpresults,
                                      qpmodel,
                                      box_constraints,
                                      dense_backend,
                                      hessian_type,
                                      QPSolver::OSQP);
  }

  // Ruiz equilibration and factorization
  ///////////////////////

  common::dense::init_qp_solve(qpsettings,
                               qpmodel,
                               qpresults,
                               qpwork,
                               box_constraints,
                               dense_backend,
                               hessian_type,
                               ruiz,
                               n_constraints,
                               QPSolver::OSQP);

  // Tmp variables
  ///////////////////////

  T primal_feasibility_eq_rhs_0(0);
  T primal_feasibility_in_rhs_0(0);
  T dual_feasibility_rhs_0(0);
  T dual_feasibility_rhs_1(0);
  T dual_feasibility_rhs_3(0);
  T primal_feasibility_lhs(0);
  T primal_feasibility_eq_lhs(0);
  T primal_feasibility_in_lhs(0);
  T dual_feasibility_lhs(0);

  T scaled_primal_feasibility_lhs(0);
  T scaled_primal_feasibility_eq_rhs_0(0);
  T scaled_primal_feasibility_in_rhs_0(0);
  T scaled_primal_feasibility_eq_lhs(0);
  T scaled_primal_feasibility_in_lhs(0);
  T scaled_dual_feasibility_lhs(0);
  T scaled_dual_feasibility_rhs_0(0);
  T scaled_dual_feasibility_rhs_1(0);
  T scaled_dual_feasibility_rhs_3(0);

  T sqrt_mu_update(0);
  T zeta_norms(0);
  T pri_res_norms(0);
  T pri_res_update(0);
  T objective_norms(0);
  T constraints_norms(0);
  T dua_res_update(0);
  T mu_update_ratio(0);

  T new_mu_eq(qpresults.info.mu_eq);
  T new_mu_in(qpresults.info.mu_in);
  T new_mu_eq_inv(qpresults.info.mu_eq_inv);
  T new_mu_in_inv(qpresults.info.mu_in_inv);

  T duality_gap(0);
  T rhs_duality_gap(0);
  T scaled_eps(qpsettings.eps_abs);

  // ADMM loop
  ///////////////////////

  for (i64 iter = 0; iter < qpsettings.max_iter; ++iter) {

    common::dense::compute_residuals(qpsettings,
                                     qpmodel,
                                     qpresults,
                                     qpwork,
                                     box_constraints,
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

    // Print iteration
    ///////////////////////

    if (qpsettings.verbose) {
      common::dense::print_iteration_line(
        qpresults, qpmodel, box_constraints, ruiz, QPSolver::OSQP, iter);
    }

    // Check if solved
    ///////////////////////

    bool can_check_termination;
    switch (qpsettings.check_solved_option) {
      case CheckSolvedStatus::ITERATION_BASED: {
        can_check_termination = true;
        break;
      }
      case CheckSolvedStatus::INTERVAL_BASED: {
        can_check_termination = qpsettings.check_termination != 0 &&
                                iter % qpsettings.check_termination == 0;
        break;
      }
    }

    if (can_check_termination) {
      bool stop_solved = common::dense::is_solved(qpsettings,
                                                  qpresults,
                                                  qpwork,
                                                  scaled_eps,
                                                  primal_feasibility_lhs,
                                                  primal_feasibility_eq_rhs_0,
                                                  primal_feasibility_in_rhs_0,
                                                  dual_feasibility_lhs,
                                                  dual_feasibility_rhs_0,
                                                  dual_feasibility_rhs_1,
                                                  dual_feasibility_rhs_3,
                                                  rhs_duality_gap);

      if (stop_solved) {
        break;
      }
    }

    // Set iteration and variables
    ///////////////////////

    qpresults.info.iter_ext += 1;

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    // ADMM step of variable updates
    ///////////////////////

    admm_step(qpsettings,
              qpmodel,
              qpresults,
              qpwork,
              box_constraints,
              n_constraints,
              dense_backend);

    // Check infeasibility
    ///////////////////////

    switch (qpsettings.check_solved_option) {
      case CheckSolvedStatus::ITERATION_BASED: {
        can_check_termination =
          iter % qpsettings.frequence_infeasibility_check == 0 ||
          qpsettings.primal_infeasibility_solving;
        break;
      }
      case CheckSolvedStatus::INTERVAL_BASED: {
        can_check_termination = qpsettings.check_termination != 0 &&
                                iter % qpsettings.check_termination == 0;
        break;
      }
    }

    if (can_check_termination) {
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
        // use active_part_z as tmp variable in order to unscale primarilly dz
        qpwork.active_part_z.tail(qpmodel.dim) = dz.tail(qpmodel.dim);
        qpwork.active_part_z.tail(qpmodel.dim).array() *=
          qpwork.i_scaled.array();
        CTdz.noalias() += qpwork.active_part_z.tail(qpmodel.dim);

        Cdx.tail(qpmodel.dim) = dx;
        Cdx.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
      }

      // compute primal and dual infeasibility criteria
      bool is_primal_infeasible =
        common::dense::global_primal_residual_infeasibility(
          VectorViewMut<T>{ from_eigen, ATdy },
          VectorViewMut<T>{ from_eigen, CTdz },
          VectorViewMut<T>{ from_eigen, dy },
          VectorViewMut<T>{ from_eigen, dz },
          qpwork,
          qpmodel,
          qpsettings,
          box_constraints,
          ruiz);

      bool is_dual_infeasible =
        common::dense::global_dual_residual_infeasibility(
          VectorViewMut<T>{ from_eigen, Adx },
          VectorViewMut<T>{ from_eigen, Cdx },
          VectorViewMut<T>{ from_eigen, Hdx },
          VectorViewMut<T>{ from_eigen, dx },
          qpwork,
          qpsettings,
          qpmodel,
          box_constraints,
          ruiz);

      if (is_primal_infeasible) {
        qpresults.info.status = QPSolverOutput::QPSOLVER_PRIMAL_INFEASIBLE;
        break;
      } else if (is_dual_infeasible) {
        qpresults.info.status = QPSolverOutput::QPSOLVER_DUAL_INFEASIBLE;
        break;
      }
    }

    // Update solver status
    ///////////////////////

    T primal_feasibility_lhs_new(primal_feasibility_lhs);
    T dual_feasibility_lhs_new(dual_feasibility_lhs);

    common::dense::compute_residuals(qpsettings,
                                     qpmodel,
                                     qpresults,
                                     qpwork,
                                     box_constraints,
                                     hessian_type,
                                     ruiz,
                                     primal_feasibility_lhs_new,
                                     primal_feasibility_eq_rhs_0,
                                     primal_feasibility_in_rhs_0,
                                     primal_feasibility_eq_lhs,
                                     primal_feasibility_in_lhs,
                                     dual_feasibility_lhs_new,
                                     dual_feasibility_rhs_0,
                                     dual_feasibility_rhs_1,
                                     dual_feasibility_rhs_3,
                                     rhs_duality_gap,
                                     duality_gap);

    common::dense::is_solved_or_closest_solved(qpsettings,
                                               qpresults,
                                               qpwork,
                                               scaled_eps,
                                               primal_feasibility_lhs_new,
                                               primal_feasibility_eq_rhs_0,
                                               primal_feasibility_in_rhs_0,
                                               dual_feasibility_lhs_new,
                                               dual_feasibility_rhs_0,
                                               dual_feasibility_rhs_1,
                                               dual_feasibility_rhs_3,
                                               rhs_duality_gap);

    // Update of proximal parameter mu
    ///////////////////////

    if (qpsettings.adaptive_mu) {
      bool iteration_condition = iter % qpsettings.adaptive_mu_interval == 0;

      if (iteration_condition) {
        scaled_global_primal_residual(
          qpmodel,
          qpresults,
          qpwork,
          box_constraints,
          scaled_primal_feasibility_lhs,      // norm(scaled pri res)
          scaled_primal_feasibility_eq_rhs_0, // norm(scaled Ax)
          scaled_primal_feasibility_in_rhs_0, // norm(scaled Cx)
          scaled_primal_feasibility_eq_lhs,   // norm(scaled Ax - zeta_eq)
          scaled_primal_feasibility_in_lhs);  // norm(scaled Cx - zeta_in)

        scaled_global_dual_residual(
          qpmodel,
          qpresults,
          qpwork,
          box_constraints,
          scaled_dual_feasibility_lhs,   // norm(scaled dua res)
          scaled_dual_feasibility_rhs_0, // norm(Hx)
          scaled_dual_feasibility_rhs_1, // norm(ATy)
          scaled_dual_feasibility_rhs_3, // norm(CTz)
          hessian_type);

        zeta_norms = std::max(infty_norm(qpresults.zeta_eq),
                              infty_norm(qpresults.zeta_in));
        pri_res_norms = std::max(scaled_primal_feasibility_eq_rhs_0,
                                 scaled_primal_feasibility_in_rhs_0);
        pri_res_update = scaled_primal_feasibility_lhs /
                         (std::max(zeta_norms, pri_res_norms) + 1e-30);

        objective_norms =
          std::max(scaled_dual_feasibility_rhs_0, infty_norm(qpwork.g_scaled));
        constraints_norms = std::max(scaled_dual_feasibility_rhs_1,
                                     scaled_dual_feasibility_rhs_3);
        dua_res_update = scaled_dual_feasibility_lhs /
                         (std::max(objective_norms, constraints_norms) + 1e-30);

        mu_update_ratio = std::sqrt(pri_res_update / dua_res_update);

        qpresults.info.rho_osqp_estimate =
          qpresults.info.mu_in_inv * mu_update_ratio;
        qpresults.info.rho_osqp_estimate = std::min(
          std::max(qpresults.info.rho_osqp_estimate, qpsettings.mu_min_in_inv),
          qpsettings.mu_max_in_inv);

        bool tolerance_condition =
          (qpresults.info.rho_osqp_estimate >
             qpresults.info.mu_in_inv * qpsettings.adaptive_mu_tolerance ||
           qpresults.info.rho_osqp_estimate <
             qpresults.info.mu_in_inv / qpsettings.adaptive_mu_tolerance);

        if (tolerance_condition) {
          {
            ++qpresults.info.mu_updates;

            new_mu_eq = 1e-3 / qpresults.info.rho_osqp_estimate;
            new_mu_in = 1.0 / qpresults.info.rho_osqp_estimate;
            new_mu_eq_inv = 1e3 * qpresults.info.rho_osqp_estimate;
            new_mu_in_inv = qpresults.info.rho_osqp_estimate;
          }
          mu_update(qpmodel,
                    qpresults,
                    qpwork,
                    n_constraints,
                    dense_backend,
                    new_mu_eq,
                    new_mu_in);

          qpresults.info.mu_eq = new_mu_eq;
          qpresults.info.mu_in = new_mu_in;
          qpresults.info.mu_eq_inv = new_mu_eq_inv;
          qpresults.info.mu_in_inv = new_mu_in_inv;
        }
      }
    }
  } // End of ADMM loop

  // Solution polishing
  ///////////////////////

  if (qpsettings.polish &&
      qpresults.info.status == QPSolverOutput::QPSOLVER_SOLVED) {

    // Timing polishing
    qpwork.timer_polish.stop();
    qpwork.timer_polish.start();

    // ADMM solution
    dense::Vec<T> x_admm = qpresults.x;
    dense::Vec<T> y_admm = qpresults.y;
    dense::Vec<T> z_admm = qpresults.z;
    dense::Vec<T> zeta_in_admm = qpresults.zeta_in;

    T pri_res_admm = qpresults.info.pri_res;
    T dua_res_admm = qpresults.info.dua_res;
    T duality_gap_admm = qpresults.info.duality_gap;

    // Find active inequality constraints (equality are considered lower-active)
    isize numactive_lower_inequalities;
    isize numactive_upper_inequalities;
    isize numactive_inequalities;
    isize inner_pb_dim;

    find_active_sets(qpsettings,
                     qpmodel,
                     qpresults,
                     qpwork,
                     box_constraints,
                     numactive_inequalities,
                     numactive_lower_inequalities,
                     numactive_upper_inequalities,
                     inner_pb_dim);

    // Build the reduced KKT matrix
    Mat<T> C_low(numactive_lower_inequalities, qpmodel.dim);
    Mat<T> C_up(numactive_upper_inequalities, qpmodel.dim);

    build_reduced_inequality_constraints_matrices(
      qpsettings, qpmodel, qpresults, qpwork, n_constraints, C_low, C_up);

    Mat<T> k_polish(inner_pb_dim, inner_pb_dim);
    Mat<T> k_plus_delta_k_polish(inner_pb_dim, inner_pb_dim);

    build_kkt_matrices_polishing(qpsettings,
                                 qpmodel,
                                 qpwork,
                                 hessian_type,
                                 k_polish,
                                 k_plus_delta_k_polish,
                                 C_low,
                                 C_up,
                                 numactive_lower_inequalities,
                                 numactive_upper_inequalities,
                                 numactive_inequalities);

    proxsuite::linalg::veg::dynstack::DynStackMut stack{
      proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_polish_stack.as_mut()
    };
    qpwork.ldl_polish.factorize(k_plus_delta_k_polish.transpose(), stack);

    // Build the reduced rhs
    Vec<T> rhs_polish(inner_pb_dim);

    build_rhs_polishing(qpsettings,
                        qpmodel,
                        qpwork,
                        hessian_type,
                        n_constraints,
                        rhs_polish,
                        numactive_lower_inequalities,
                        numactive_upper_inequalities);

    // Solve K t = rhs before iterative refinement
    Vec<T> hat_t = rhs_polish;

    qpwork.ldl_polish.solve_in_place(hat_t.head(inner_pb_dim), stack);

    // Iterative refinement
    Vec<T> rhs_polish_refine(inner_pb_dim);
    Vec<T> delta_hat_t(inner_pb_dim);

    for (i64 iter = 0; iter < qpsettings.polish_refine_iter; ++iter) {
      rhs_polish_refine = rhs_polish - k_polish * hat_t;
      delta_hat_t = rhs_polish_refine;

      qpwork.ldl_polish.solve_in_place(delta_hat_t.head(inner_pb_dim), stack);

      hat_t = hat_t + delta_hat_t;
    }

    // Update variables
    update_variables_polishing(qpmodel,
                               qpresults,
                               qpwork,
                               box_constraints,
                               n_constraints,
                               hat_t,
                               numactive_lower_inequalities);

    // Check if solution polishing succeeded
    common::dense::global_primal_residual(qpmodel,
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

    common::dense::global_dual_residual(qpresults,
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

    bool polish_succeeded =
      (qpresults.info.pri_res < pri_res_admm &&
       qpresults.info.dua_res < dua_res_admm) ||
      (qpresults.info.pri_res < pri_res_admm && dua_res_admm < 1e-10) ||
      (qpresults.info.dua_res < dua_res_admm && pri_res_admm < 1e-10);

    if (polish_succeeded) {
      qpresults.info.status_polish = PolishStatus::POLISH_SUCCEEDED;
    } else {
      qpresults.x = x_admm;
      qpresults.y = y_admm;
      qpresults.z = z_admm;
      qpresults.zeta_in = zeta_in_admm;

      qpresults.info.pri_res = pri_res_admm;
      qpresults.info.dua_res = dua_res_admm;
      qpresults.info.duality_gap = duality_gap_admm;

      qpresults.info.status_polish = PolishStatus::POLISH_FAILED;
    }

    // Timing polishing
    qpresults.info.polish_time = qpwork.timer_polish.elapsed().user;

    // Print polishing info
    if (qpsettings.verbose) {
      print_polishing_line(qpsettings, qpresults);
    }
  }

  // End of qp_solve
  ///////////////////////

  common::dense::unscale_solver(
    qpsettings, qpmodel, qpresults, box_constraints, ruiz);

  common::dense::compute_objective(qpresults, qpmodel);

  if (qpsettings.compute_timings) {
    common::dense::compute_timings(qpresults, qpwork);
  }

  if (qpsettings.verbose) {
    common::dense::print_solver_statistics(
      qpsettings, qpresults, QPSolver::OSQP);
  }

  qpwork.dirty = true;
  qpwork.is_initialized = true; // necessary because we call workspace cleanup

  assert(!std::isnan(qpresults.info.pri_res));
  assert(!std::isnan(qpresults.info.dua_res));
  assert(!std::isnan(qpresults.info.duality_gap));

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */
