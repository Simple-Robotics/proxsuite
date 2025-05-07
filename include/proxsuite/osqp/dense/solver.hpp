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
admm_iter(const Settings<T>& qpsettings,
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
  Vec<T> zeta_tilde_eq;

  qpwork.rhs.setZero();
  qpwork.rhs.head(qpmodel.dim) =
    qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
    qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;

  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq;
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

  // Update the variables
  zeta_tilde_eq =
    qpwork.b_scaled + qpresults.info.mu_eq * (nu_eq - qpresults.y);
  qpresults.x =
    qpsettings.alpha_osqp * x_tilde + (1 - qpsettings.alpha_osqp) * qpresults.x;
  qpresults.y = qpresults.y + qpresults.info.mu_eq_inv * qpsettings.alpha_osqp *
                                (zeta_tilde_eq - qpwork.b_scaled);
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

    bool stop_loop = false;
    proxsuite::common::compute_feasibility(qpsettings,
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
                                           iter,
                                           stop_loop);
    if (stop_loop) {
      break;
    }

    qpresults.info.iter_ext += 1;

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// 1

    // Activate sets, specific to inequality

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// 2

    admm_iter(qpsettings,
              qpmodel,
              qpresults,
              qpwork,
              box_constraints,
              n_constraints,
              ruiz,
              dense_backend,
              hessian_type);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// 3

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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// 4

    // Update of mu

  } // End of loop of ADMM iterations

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 5

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