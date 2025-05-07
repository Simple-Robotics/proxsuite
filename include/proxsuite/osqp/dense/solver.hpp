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
  }

  qpresults.info.iter_ext += 1;

  qpwork.x_prev = qpresults.x;
  qpwork.y_prev = qpresults.y;
  qpwork.z_prev = qpresults.z;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 1

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 2

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 3

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 4

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 5

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 6

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 7

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// 8

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */