//
// Copyright (c) 2022 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_SOLVER_HPP
#define PROXSUITE_OSQP_DENSE_SOLVER_HPP

#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/dense/preconditioner/ruiz.hpp"
#include "proxsuite/solvers/common/utils.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include <iostream>

namespace proxsuite {
namespace osqp {
namespace dense {

/*!
 * Solves the KKT system in the case of equality constraints only.
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
solve_eq_constraints_system(const proxqp::Settings<T>& qpsettings,
                            const proxqp::dense::Model<T>& qpmodel,
                            proxqp::Results<T>& qpresults,
                            proxqp::dense::Workspace<T>& qpwork,
                            const bool box_constraints,
                            proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
                            const proxqp::DenseBackend dense_backend,
                            const proxqp::HessianType hessian_type)
{

  using namespace proxsuite::proxqp;

  // Solve the linear system


  // Check the feasibility (to be done after the solve)
  // Note: Functions not recognized as the arguments are not defined yet 
  // -> first solve the system, then define the arguments

  auto& Hdx = qpwork.Hdx;
  auto& Adx = qpwork.Adx;
  auto& ATdy = qpwork.CTz;

  if (qpsettings.primal_infeasibility_solving) {
    // compute primal and dual infeasibility criteria
    bool is_primal_infeasible = proxsuite::proxqp::dense::global_primal_residual_infeasibility(
      VectorViewMut<T>{ from_eigen, ATdy },
      VectorViewMut<T>{ from_eigen, ATdy - ATdy }, // Put to zero // TODO: Learn how the type works
      VectorViewMut<T>{ from_eigen, ATdy - ATdy },   // Just y - y_prev // TODO: Compute then do the difference
      VectorViewMut<T>{ from_eigen, ATdy - ATdy },   // Put to zero // TODO: Learn how the type works
      qpwork,
      qpmodel,
      qpsettings,
      box_constraints,
      ruiz);

    bool is_dual_infeasible = proxsuite::proxqp::dense::global_dual_residual_infeasibility(
      VectorViewMut<T>{ from_eigen, Adx },
      VectorViewMut<T>{ from_eigen, ATdy - ATdy }, // Put to zero // TODO: Learn how the type works
      VectorViewMut<T>{ from_eigen, Hdx },
      VectorViewMut<T>{ from_eigen, ATdy - ATdy }, // Just x - x_prev // TODO: Compute then do the difference
      qpwork,
      qpsettings,
      qpmodel,
      box_constraints,
      ruiz);

    if (is_primal_infeasible) {
      qpresults.info.status = QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE;
      if (qpsettings.primal_infeasibility_solving) {
        qpresults.info.status = QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE; // My idea => check if OK
      }
    } else if (is_dual_infeasible) {
      qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
    }
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
  const proxqp::Settings<T>& qpsettings,
  const proxqp::dense::Model<T>& qpmodel,
  proxqp::Results<T>& qpresults,
  proxqp::dense::Workspace<T>& qpwork,
  const bool box_constraints,
  const proxqp::DenseBackend& dense_backend,
  const proxqp::HessianType& hessian_type,
  proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz)
{
  PROXSUITE_EIGEN_MALLOC_NOT_ALLOWED();

  // Initialization

  proxsuite::solvers::utils::setup_solver(qpsettings,
                                           qpmodel,
                                           qpresults,
                                           qpwork,
                                           box_constraints,
                                           dense_backend,
                                           hessian_type,
                                           ruiz,
                                           proxsuite::solvers::utils::SolverType::OSQP);

  // Solve
  std::cout << "OSQP solver" << std::endl;

  // Variables for stopping criterion

  T primal_feasibility_eq_rhs_0(0); // norm(unscaled(Ax))
  T primal_feasibility_in_rhs_0(0); // norm(unscaled(Cx))
  T dual_feasibility_rhs_0(0);      // norm(unscaled(Hx))
  T dual_feasibility_rhs_1(0);      // norm(unscaled(ATy))  
  T dual_feasibility_rhs_3(0);      // norm(unscaled(CTz))
  T primal_feasibility_eq_lhs(0);   // norm(unscaled(Ax - b))
  T primal_feasibility_in_lhs(0);   // norm(unscaled([Cx - u]+ + [Cx - l]-))
  T primal_feasibility_lhs(0);      // max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs)
  T dual_feasibility_lhs(0);        // norm(scaled(Hx + g + ATy + CTz))    
                                    // TODO: Understand why one is scaled and the other is not
  T duality_gap(0);                 // xHx + gTx + bTy + zTu + zTl
  T rhs_duality_gap(0);             // max(abs(each term))
  T scaled_eps(qpsettings.eps_abs); // eps_abs

  for (proxqp::i64 iter = 0; iter < qpsettings.max_iter; ++iter) {

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // General
  
    bool stopping_criteria = proxsuite::solvers::utils::compute_residuals_and_infeasibility_1(qpsettings, 
                                                    qpmodel, 
                                                    qpresults, 
                                                    qpwork, 
                                                    box_constraints, 
                                                    hessian_type, 
                                                    ruiz, 
                                                    proxsuite::solvers::utils::SolverType::OSQP, 
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
    
    if (stopping_criteria) {
      break;
    }

    qpresults.info.iter_ext += 1; // We start a new external loop update

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Specific: Corpus of the solver

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // General

    T primal_feasibility_lhs_new(primal_feasibility_lhs);


    proxsuite::solvers::utils::compute_residuals_and_infeasibility_2(qpsettings, 
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

  }

  // End

  proxsuite::solvers::utils::unscale_solver(qpsettings, 
                                            qpmodel,
                                            qpresults, 
                                            box_constraints, 
                                            ruiz);

  proxsuite::solvers::utils::compute_objective(qpmodel, 
                                               qpresults);

  proxsuite::solvers::utils::compute_timings(qpsettings, 
                                            qpresults, 
                                            qpwork);

  proxsuite::solvers::utils::print_solver_statistics(qpsettings, 
                                                     qpresults, 
                                                     proxsuite::solvers::utils::SolverType::OSQP);

  proxsuite::solvers::utils::prepare_next_solve(qpresults, 
                                                qpwork);

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense

} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */
