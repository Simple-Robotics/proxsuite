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
#include <proxsuite/linalg/dense/ldlt.hpp>
#include "proxsuite/proxqp/dense/solver.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include "proxsuite/proxqp/dense/fwd.hpp"
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
bool
solve_eq_constraints_system(const proxqp::Settings<T>& qpsettings,
                            const proxqp::dense::Model<T>& qpmodel,
                            proxqp::Results<T>& qpresults,
                            proxqp::dense::Workspace<T>& qpwork,
                            const bool box_constraints,
                            const proxqp::isize n_constraints,
                            proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
                            const proxqp::DenseBackend dense_backend,
                            const proxqp::HessianType hessian_type)
{

  using namespace proxsuite::proxqp;

  // Solve the linear system
  // ( (H + rho * I, A^T)  (A, -\mu_eq * I) ) ( x, y ) = ( rho * x_prev - g, b - \mu_eq * y_prev ) 
  // Formulation in proxqp paper, equality case

  qpwork.rhs.setZero();
  qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.tail(qpmodel.n_eq) = qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;

  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

  proxqp::dense::solve_linear_system(qpwork.rhs,
                                     qpmodel,
                                     qpresults,
                                     qpwork,
                                     n_constraints,
                                     dense_backend,
                                     inner_pb_dim,
                                     stack);

  /// New (x, y)
  qpresults.x = qpwork.rhs.head(qpmodel.dim);
  qpresults.y = qpwork.rhs.tail(qpmodel.n_eq);

  // Delta x and y (and z equal to 0)
  proxqp::dense::Vec<T> dx = qpresults.x - qpwork.x_prev; // TODO: Dirty way to define it ? (Compared to proxq code, line "auto dx = qpwork.dw_aug.head(qpmodel.dim);")
  proxqp::dense::Vec<T> dy = qpresults.y - qpwork.y_prev;
  proxqp::dense::Vec<T> dz = proxqp::dense::Vec<T>::Zero(qpmodel.n_in);

  // New right hand side 
  qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.tail(qpmodel.n_eq) = qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;


  // // Computing the resuduals: 
  // // Note: The linear system in proxqp solves finds dx, dy and so on => I change the definitions
  // // of Adx, etc (comapred to proxqp solver)

  auto& Hdx = qpwork.Hdx;
  auto& Adx = qpwork.Adx;
  auto& ATdy = qpwork.CTz;

  switch (hessian_type) {
    case HessianType::Zero:
      break;
    case HessianType::Dense:
      Hdx.noalias() = qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * dx;
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

  // proxsuite::linalg::veg::dynstack::DynStackMut stack{
  //   proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  // }; // TODO: See how to deal with the stack, how it works
  auto& Cdx = qpwork.Cdx;
  LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
  if (qpmodel.n_in > 0) {
    Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
    CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
  }

  // Check the feasibility (code inspired from the proxqp newton function)

  if (qpsettings.primal_infeasibility_solving) {
    bool is_primal_infeasible = proxsuite::proxqp::dense::global_primal_residual_infeasibility(
      VectorViewMut<T>{ from_eigen, ATdy },
      VectorViewMut<T>{ from_eigen, CTdz  },
      VectorViewMut<T>{ from_eigen, dy },   
      VectorViewMut<T>{ from_eigen, dz },
      qpwork,
      qpmodel,
      qpsettings,
      box_constraints,
      ruiz);

    bool is_dual_infeasible = proxsuite::proxqp::dense::global_dual_residual_infeasibility(
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
      qpresults.info.status = QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE;
    } else if (is_dual_infeasible) {
      qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
    }
  }

  if ((qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE &&
        !qpsettings.primal_infeasibility_solving) ||
      qpresults.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
    // certificate of infeasibility
    return true; // Instead of doing break; in one single big qp_solve function
  } else {
    return false;
  }

  // TODO: See more in detail on OSQP algorithm how I implement the 
  // solve closest problem solution (if infeasible)

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

  proxqp::isize n_constraints(qpmodel.n_in);

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

    bool stopping_criteria_bis = solve_eq_constraints_system(qpsettings, // TODO: Is it well written ? (both solve and boolean or stopping)
                                qpmodel, 
                                qpresults, 
                                qpwork, 
                                box_constraints, 
                                n_constraints,
                                ruiz, 
                                dense_backend, 
                                hessian_type);

    if (stopping_criteria_bis) {
      break;
    }

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
