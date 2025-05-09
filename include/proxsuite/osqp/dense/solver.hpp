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
#include "proxsuite/solvers/common/utils.hpp"
#include <iostream>

namespace proxsuite {
namespace osqp {
namespace dense {

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::dense;

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
 * @param ruiz ruiz preconditioner.
 */
template<typename T>
void
admm_step(const Settings<T>& qpsettings,
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
  T scaled_eps(qpsettings.eps_abs);

  for (i64 iter = 0; iter < qpsettings.max_iter; ++iter) {

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
                                   iter);
    if (is_solved_qp) {
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
              ruiz,
              dense_backend,
              hessian_type);

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

    T primal_feasibility_lhs_new(primal_feasibility_lhs);
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
                                            dual_feasibility_lhs,
                                            dual_feasibility_rhs_0,
                                            dual_feasibility_rhs_1,
                                            dual_feasibility_rhs_3,
                                            rhs_duality_gap,
                                            duality_gap,
                                            scaled_eps);

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// mu update

    //////////////////////////////////////////////////////////////////////////////////////////////
    /// end of mu update
  }

  proxsuite::common::unscale_solver(
    qpsettings, qpmodel, qpresults, box_constraints, ruiz);
  proxsuite::common::compute_objective(qpmodel, qpresults);
  if (qpsettings.compute_timings) {
    proxsuite::common::compute_timings(qpsettings, qpresults, qpwork);
  }

  if (qpsettings.verbose) {
    proxsuite::common::print_solver_statistics(
      qpsettings, qpresults, common::QPSolver::PROXQP);
  }

  proxsuite::common::prepare_next_solve(qpresults, qpwork);

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */