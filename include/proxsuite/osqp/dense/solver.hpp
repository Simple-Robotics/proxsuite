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

/*!
 * Derives the global primal residual of the QP problem.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param scaled_primal_feasibility_lhs scaled primal infeasibility.
 * @param scaled_primal_feasibility_eq_rhs_0 scaled scalar variable used when using a relative
 * stopping criterion.
 * @param scaled_primal_feasibility_in_rhs_0 scaled scalar variable used when using a relative
 * stopping criterion.
 * @param scaled_primal_feasibility_eq_lhs scaled scalar variable used when using a relative
 * stopping criterion.
 * @param scaled_primal_feasibility_in_lhs scaled scalar variable used when using a relative
 * stopping criterion.
 */
template<typename T>
void
scaled_global_primal_residual(const proxqp::dense::Model<T>& qpmodel,
                       proxqp::Results<T>& qpresults,
                       proxqp::dense::Workspace<T>& qpwork,
                       const proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
                       const bool box_constraints,
                       T& scaled_primal_feasibility_lhs,
                       T& scaled_primal_feasibility_eq_rhs_0,
                       T& scaled_primal_feasibility_in_rhs_0,
                       T& scaled_primal_feasibility_eq_lhs,
                       T& scaled_primal_feasibility_in_lhs)
{

  // Function inspired from global_primal_residual, but:
  // We compute the scaled terms
  // We add the norm of zeta (z in osqp paper) in the max for the primal residual criteria

  using namespace proxsuite::proxqp;

  qpresults.se.noalias() = qpwork.A_scaled * qpresults.x;                                
  scaled_primal_feasibility_eq_rhs_0 = proxqp::dense::infty_norm(qpresults.se);                        

  qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in).noalias() =                     
    qpwork.C_scaled * qpresults.x;
  if (box_constraints) {
    qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) = qpresults.x;                 
  }
  scaled_primal_feasibility_in_rhs_0 =
    proxqp::dense::infty_norm(qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in));                 

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
               proxqp::dense::infty_norm(qpwork.active_part_z.tail(qpmodel.dim)));
    scaled_primal_feasibility_in_rhs_0 =
      std::max(scaled_primal_feasibility_in_rhs_0, proxqp::dense::infty_norm(qpresults.x));    
  }

  
  ruiz.unscale_primal_residual_in_place_eq(                                          
    VectorViewMut<T>{ from_eigen, qpresults.se });
  qpresults.se -= qpmodel.b;   
  ruiz.scale_primal_residual_in_place_eq(
    VectorViewMut<T>{ from_eigen, qpresults.se });                                   
  scaled_primal_feasibility_eq_lhs = proxqp::dense::infty_norm(qpresults.se);                 


  scaled_primal_feasibility_in_lhs = proxqp::dense::infty_norm(qpresults.si);                    
  ruiz.unscale_primal_residual_in_place_in(
    VectorViewMut<T>{ from_eigen, qpresults.si.head(qpmodel.n_in) });                
  if (box_constraints){
    ruiz.unscale_primal_residual_in_place(
    VectorViewMut<T>{ from_eigen, qpresults.si.tail(qpmodel.dim) });                    
  }
  
  scaled_primal_feasibility_lhs =
    std::max(scaled_primal_feasibility_eq_lhs, scaled_primal_feasibility_in_lhs);

}
/*!
 * Derives the global dual residual of the QP problem.
 *
 * @param qpwork solver workspace.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param dual_feasibility_lhs primal infeasibility.
 * @param primal_feasibility_eq_rhs_0 scaled scalar variable used when using a relative
 * stopping criterion.
 * @param dual_feasibility_rhs_0 scaled scalar variable used when using a relative
 * stopping criterion.
 * @param dual_feasibility_rhs_1 scaled scalar variable used when using a relative
 * stopping criterion.
 * @param dual_feasibility_rhs_3 scaled scalar variable used when using a relative
 * stopping criterion.
 */
template<typename T>
void
scaled_global_dual_residual(proxqp::Results<T>& qpresults,
                     proxqp::dense::Workspace<T>& qpwork,
                     const proxqp::dense::Model<T>& qpmodel,
                     const bool box_constraints,
                     T& scaled_dual_feasibility_lhs,
                     T& scaled_dual_feasibility_rhs_0,
                     T& scaled_dual_feasibility_rhs_1,
                     T& scaled_dual_feasibility_rhs_3,
                     const proxqp::HessianType& hessian_type)
{
  
  using namespace proxsuite::proxqp;


  qpwork.dual_residual_scaled = qpwork.g_scaled;                                   

  switch (hessian_type) {
    case HessianType::Zero:
      scaled_dual_feasibility_rhs_0 = 0;
      break;
    case HessianType::Dense:
      qpwork.CTz.noalias() =
        qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * qpresults.x;
      qpwork.dual_residual_scaled += qpwork.CTz;
      scaled_dual_feasibility_rhs_0 = proxqp::dense::infty_norm(qpwork.CTz);  
      break;
    case HessianType::Diagonal:
      qpwork.CTz.array() =
        qpwork.H_scaled.diagonal().array() * qpresults.x.array();
      qpwork.dual_residual_scaled += qpwork.CTz;
      scaled_dual_feasibility_rhs_0 = proxqp::dense::infty_norm(qpwork.CTz);  
      break;
  }                                                                                

  qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * qpresults.y;
  qpwork.dual_residual_scaled += qpwork.CTz;                                       
  scaled_dual_feasibility_rhs_1 = proxqp::dense::infty_norm(qpwork.CTz);                           

  qpwork.CTz.noalias() =
    qpwork.C_scaled.transpose() * qpresults.z.head(qpmodel.n_in);
  qpwork.dual_residual_scaled += qpwork.CTz;                                                           
  scaled_dual_feasibility_rhs_3 = proxqp::dense::infty_norm(qpwork.CTz);
  if (box_constraints) {
    qpwork.CTz.noalias() = qpresults.z.tail(qpmodel.dim);
    qpwork.CTz.array() *= qpwork.i_scaled.array();

    qpwork.dual_residual_scaled += qpwork.CTz;
    scaled_dual_feasibility_rhs_3 =
      std::max(proxqp::dense::infty_norm(qpwork.CTz), scaled_dual_feasibility_rhs_3);
  }                                                                                  

  scaled_dual_feasibility_lhs = proxqp::dense::infty_norm(qpwork.dual_residual_scaled);              

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
compute_update_mu_criteria(proxqp::dense::Workspace<T>& qpwork,
                          const proxqp::dense::Model<T>& qpmodel,
                          proxqp::Results<T>& qpresults,
                          const proxqp::HessianType& hessian_type,
                          const proxqp::dense::preconditioner::RuizEquilibration<T> &ruiz,
                          const bool box_constraints,
                          T &scaled_primal_feasibility_lhs, 
                          T &scaled_primal_feasibility_eq_rhs_0, 
                          T &scaled_primal_feasibility_in_rhs_0, 
                          T &scaled_primal_feasibility_eq_lhs, 
                          T &scaled_primal_feasibility_in_lhs,
                          T &scaled_dual_feasibility_lhs,
                          T &scaled_dual_feasibility_rhs_0,
                          T &scaled_dual_feasibility_rhs_1,
                          T &scaled_dual_feasibility_rhs_3)
{

  using namespace proxsuite::proxqp;

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

  T epsilon = 1e-9; // TODO: Better way to code this (to avoid deviding by zero) ?

  T primal_term = (scaled_primal_feasibility_lhs + epsilon) / (std::max(scaled_primal_feasibility_eq_rhs_0,
                                                               std::max(proxqp::dense::infty_norm(qpwork.zeta_eq), 
                                                               std::max(proxqp::dense::infty_norm(qpwork.zeta_in), 
                                                               scaled_primal_feasibility_in_rhs_0))) 
                                                               + epsilon);

  T dual_term = (scaled_dual_feasibility_lhs + epsilon) / (std::max(scaled_dual_feasibility_rhs_0, 
                                                           std::max(scaled_dual_feasibility_rhs_1, 
                                                           std::max(proxqp::dense::infty_norm(qpwork.g_scaled),
                                                           scaled_dual_feasibility_rhs_3))) 
                                                           + epsilon);

  T update_mu_criteria = std::sqrt((primal_term + epsilon) / (dual_term + epsilon));

  // TODO: See how to manage the case where the dual term is very little (add some epsilon somewhere ?)

  return update_mu_criteria;

}
/*!
 * Setups and performs the factorization of the full regularized KKT matrix of
 * the problem (containing equality and inequality constraints).
 *
 * @param qpwork workspace of the solver.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solution results.
 */
template<typename T>
void
setup_factorization_full_kkt(proxqp::dense::Workspace<T>& qpwork,
                             const proxqp::dense::Model<T>& qpmodel,
                             proxqp::Results<T>& qpresults,
                             const proxqp::DenseBackend dense_backend,
                             const proxqp::dense::isize n_constraints)
{

  using namespace proxsuite::proxqp;

  isize n_c_f = qpwork.n_c; // 0 at initialization
  
  // Start from the equality KKT, then add the inequality and box constraints

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

  {
    auto _planned_to_add = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, n_constraints);
    auto planned_to_add = _planned_to_add.ptr_mut();

    isize planned_to_add_count = 0;
    T mu_in_neg(-qpresults.info.mu_in);

    isize n_c = n_c_f;
    for (isize i = 0; i < n_constraints; i++) {
      planned_to_add[planned_to_add_count] = i;
      ++planned_to_add_count;
      n_c_f += 1;
    }
    {
      switch (dense_backend) {
        case DenseBackend::PrimalDualLDLT: {
          isize n = qpmodel.dim;
          isize n_eq = qpmodel.n_eq;
          LDLT_TEMP_MAT_UNINIT(
            T, new_cols, n + n_eq + n_c_f, planned_to_add_count, stack);
          for (isize k = 0; k < planned_to_add_count; ++k) {
            isize index = planned_to_add[k];
            auto col = new_cols.col(k);
            col.head(n) = (qpwork.C_scaled.row(index));
            // Add the box constraints too 
            // if (index >= qpmodel.n_in) {
            //   col.head(n).setZero();
            //   col[index - qpmodel.n_in] = qpwork.i_scaled[index - qpmodel.n_in];
            // } else {
            //   col.head(n) = (qpwork.C_scaled.row(index));
            // }
            col.tail(n_eq + n_c_f).setZero();
            col[n + n_eq + n_c + k] = mu_in_neg;
          }
          qpwork.timer_full_fact.start();
          qpwork.ldl.insert_block_at(n + n_eq + n_c, new_cols, stack);
          qpwork.timer_full_fact.stop();
          qpwork.factorization_time_full_kkt = qpwork.timer_full_fact.elapsed().user;
        } break;
        case DenseBackend::PrimalLDLT: { // TODO: Code the PrimalLDLT case
        } break;
        case DenseBackend::Automatic: {
        } break;
      }
    }
  }

  qpwork.n_c = n_c_f;

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
check_infeasibility(const proxqp::Settings<T>& qpsettings,
                    const proxqp::dense::Model<T>& qpmodel,
                    proxqp::Results<T>& qpresults,
                    proxqp::dense::Workspace<T>& qpwork,
                    const bool box_constraints,
                    const proxqp::isize n_constraints,
                    proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
                    const proxqp::HessianType hessian_type,
                    const proxsuite::linalg::veg::dynstack::DynStackMut stack)
{

  using namespace proxsuite::proxqp;

  // Check the infeasibility

  // Delta x, y and z
  proxqp::dense::Vec<T> dx = qpresults.x - qpwork.x_prev;
  proxqp::dense::Vec<T> dy = qpresults.y - qpwork.y_prev;
  proxqp::dense::Vec<T> dz = qpresults.z - qpwork.z_prev;

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

  auto& Cdx = qpwork.Cdx;
  LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
  if (qpmodel.n_in > 0) {
    Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
    CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
  }

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

  if ((qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE &&
        !qpsettings.primal_infeasibility_solving) ||
      qpresults.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
    return true; // Instead of doing break; in one single big qp_solve function
  } else {
    return false;
  }

  // TODO: Code for the closest feasible primal solution in case of primal infeasibility


}
/*!
 * Iteration of ADMM step.
 * Solves the KKT system with equality and inequality constraints, then update the primal and dual variables
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
admm(const proxqp::Settings<T>& qpsettings,
     const proxqp::dense::Model<T>& qpmodel,
     proxqp::Results<T>& qpresults,
     proxqp::dense::Workspace<T>& qpwork,
     const bool box_constraints,
     const proxqp::isize n_constraints,
     proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
     const proxqp::DenseBackend dense_backend,
     const proxqp::HessianType hessian_type)
{

  // Implementation note: 
  // It is convenient, given the architecture of proxsuite, to reuse the same functions (infeasibility, residuals, etc)
  // and wrapper than the proxqp solver. Given the different problem formulation 
  // (see https://inria.hal.science/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf/), the variables 
  // y and z in osqp are adapted to our framework, namely with zeta_eq = z_osqp[eq:], zeta_in = z_osqp[:in] and y_osqp = concat(y_proxqp, z_proxqp).


  using namespace proxsuite::proxqp;

  // Corpus of the ADMM iteration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // Solves the linear system ((24) in https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf with the proxqp wrapper / framework)

  // Print mu_eq and mu_in
  // std::cout << "mu_eq: " << qpresults.info.mu_eq << std::endl;
  // std::cout << "mu_in: " << qpresults.info.mu_in << std::endl;

  qpwork.rhs.setZero();
  qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;
  qpwork.rhs.tail(qpmodel.n_in) = qpwork.zeta_in - qpresults.info.mu_in * qpresults.z;

  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + qpmodel.n_in;
  
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

  // Updates of x, y, z (according to OSQP)

  // Solution of the linear system
  proxqp::dense::Vec<T> x_tilde = qpwork.rhs.head(qpmodel.dim);
  proxqp::dense::Vec<T> nu_eq = qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq);
  proxqp::dense::Vec<T> nu_in = qpwork.rhs.tail(qpmodel.n_in);

  // Update of x
  qpresults.x = qpsettings.alpha_osqp * x_tilde + (1 - qpsettings.alpha_osqp) * qpresults.x;

  // Update of the zeta_tilde's
  proxqp::dense::Vec<T> zeta_tilde_eq = qpwork.zeta_eq + qpresults.info.mu_eq * (nu_eq - qpresults.y);
  proxqp::dense::Vec<T> zeta_tilde_in = qpwork.zeta_in + qpresults.info.mu_in * (nu_in - qpresults.z);

  // Update of the zeta's (orthogonal projection on the constraints)
  proxqp::dense::Vec<T> new_zeta_eq = qpwork.b_scaled;
  proxqp::dense::Vec<T> sum_zeta_in_z = qpsettings.alpha_osqp * zeta_tilde_in 
                                      + (1 - qpsettings.alpha_osqp) * qpwork.zeta_in 
                                      + qpresults.info.mu_in * qpresults.z;
  proxqp::dense::Vec<T> new_zeta_in = qpwork.l_scaled.cwiseMax(sum_zeta_in_z.cwiseMin(qpwork.u_scaled));
    

  // Update of y and z
  qpresults.y = qpresults.y + qpresults.info.mu_eq_inv * (qpsettings.alpha_osqp * zeta_tilde_eq + (1 - qpsettings.alpha_osqp) * qpwork.zeta_eq - new_zeta_eq);
  qpresults.z = qpresults.z + qpresults.info.mu_in_inv * (qpsettings.alpha_osqp * zeta_tilde_in + (1 - qpsettings.alpha_osqp) * qpwork.zeta_in - new_zeta_in);

  // Update of the zeta's
  qpwork.zeta_eq = new_zeta_eq;
  qpwork.zeta_in = new_zeta_in;

  // New right hand side
  qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.zeta_eq - qpresults.info.mu_eq * qpresults.y;
  qpwork.rhs.tail(qpmodel.n_in) = qpwork.zeta_in - qpresults.info.mu_in * qpresults.z;

  // Prints x, y, z
  // std::cout << "x: " << qpresults.x << std::endl;
  // std::cout << "y: " << qpresults.y << std::endl;
  // std::cout << "z: " << qpresults.z << std::endl;

  bool infeas_check = check_infeasibility(qpsettings, 
                                          qpmodel,
                                          qpresults, 
                                          qpwork, 
                                          box_constraints, 
                                          n_constraints, 
                                          ruiz,
                                          hessian_type, 
                                          stack);

  return infeas_check;

}
/*!
 * Polishing step after the ADMM.
 * Iterative refinement of the solution from the system K + Delta_K * hat_t = rhs.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
polish(const proxqp::Settings<T>& qpsettings,
     const proxqp::dense::Model<T>& qpmodel,
     proxqp::Results<T>& qpresults,
     proxqp::dense::Workspace<T>& qpwork,
     const bool box_constraints,
     const proxqp::isize n_constraints,
     const proxqp::DenseBackend dense_backend,
     const proxqp::HessianType hessian_type)
{

  using namespace proxsuite::proxqp;

  // TODO: Code for PrimalLDLT backend
  (void)box_constraints; // TODO: Code the box constraints

  // Prints
  // std::cout << "Variables before polishing: " << std::endl;
  // std::cout << "x: " << qpresults.x << std::endl;
  // std::cout << "y: " << qpresults.y << std::endl;
  // std::cout << "z: " << qpresults.z << std::endl;

  // Find the upper and lower active constraints
  qpwork.active_set_up_eq.array() = (qpresults.y.array() > 0); 
  qpwork.active_set_low_eq.array() = (qpresults.y.array() < 0); 
  qpwork.active_constraints_eq = qpwork.active_set_up_eq || qpwork.active_set_low_eq; 
  isize numactive_constraints_eq = qpwork.active_constraints_eq.count();
  isize numactive_constraints_eq_up = qpwork.active_set_up_eq.count();
  isize numactive_constraints_eq_low = qpwork.active_set_low_eq.count();

  // qpwork.active_set_up_ineq.array() = (qpresults.z.array() > 0); // To be changed
  // qpwork.active_set_low_ineq.array() = (qpresults.z.array() < 0); // To be changed
  qpwork.active_set_up_ineq.array() = (qpresults.z.array() > (qpwork.u_scaled - qpwork.zeta_in).array()); 
  qpwork.active_set_low_ineq.array() = (qpresults.z.array() < -(qpwork.zeta_in - qpwork.l_scaled).array()); 
  qpwork.active_constraints_ineq = qpwork.active_set_up_ineq || qpwork.active_set_low_ineq; 
  isize numactive_constraints_ineq = qpwork.active_constraints_ineq.count();
  isize numactive_constraints_ineq_up = qpwork.active_set_up_ineq.count();
  isize numactive_constraints_ineq_low = qpwork.active_set_low_ineq.count();

  isize numactive_constraints = numactive_constraints_eq + numactive_constraints_ineq;

  // Construction of K
  qpwork.k_polish.resize(qpmodel.dim + numactive_constraints, qpmodel.dim + numactive_constraints);
  switch (hessian_type) {
    case HessianType::Dense:
      qpwork.k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
      break;
    case HessianType::Zero:
      qpwork.k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim).setZero();
      break;
    case HessianType::Diagonal:
      qpwork.k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
      break;
  }
  
  proxqp::dense::Mat<T> C_low(numactive_constraints_ineq_low, qpmodel.dim);
  proxqp::dense::Mat<T> C_up(numactive_constraints_ineq_up, qpmodel.dim);

  isize low_index = 0;
  isize up_index = 0;
  for (isize i = 0; i < qpmodel.n_in; ++i) {
    if (qpwork.active_set_low_ineq(i)) {
      C_low.row(low_index) = qpwork.C_scaled.row(i);
      ++low_index;
    }
    if (qpwork.active_set_up_ineq(i)) {
      C_up.row(up_index) = qpwork.C_scaled.row(i);
      ++up_index;
    }
  }

  proxqp::dense::Mat<T> A_low(numactive_constraints_eq_low, qpmodel.dim);
  proxqp::dense::Mat<T> A_up(numactive_constraints_eq_up, qpmodel.dim);

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

  // Here, define the matrix properly with A_L, A_U, C_L, C_U
  isize row = qpmodel.dim;
  isize column = qpmodel.dim;
  qpwork.k_polish.block(0, column, qpmodel.dim, numactive_constraints_eq_low) = A_low.transpose();
  column += numactive_constraints_eq_low;
  qpwork.k_polish.block(0, column, qpmodel.dim, numactive_constraints_ineq_low) = C_low.transpose();
  column += numactive_constraints_ineq_low;
  qpwork.k_polish.block(0, column, qpmodel.dim, numactive_constraints_eq_up) = A_up.transpose();
  column += numactive_constraints_eq_up;
  qpwork.k_polish.block(0, column, qpmodel.dim, numactive_constraints_ineq_up) = C_up.transpose();
  qpwork.k_polish.block(row, 0, numactive_constraints_eq_low, qpmodel.dim) = A_low;
  row+= numactive_constraints_eq_low;
  qpwork.k_polish.block(row, 0, numactive_constraints_ineq_low, qpmodel.dim) = C_low;
  row+= numactive_constraints_ineq_low;
  qpwork.k_polish.block(row, 0, numactive_constraints_eq_up, qpmodel.dim) = A_up;
  row+= numactive_constraints_eq_up;
  qpwork.k_polish.block(row, 0, numactive_constraints_ineq_up, qpmodel.dim) = C_up;
  qpwork.k_polish.bottomRightCorner(numactive_constraints, numactive_constraints).setZero();

  // Construction and factorization of K + Delta_K
  qpwork.k_plus_delta_k_polish = qpwork.k_polish;
  qpwork.k_plus_delta_k_polish.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() += qpsettings.delta_osqp; 
  qpwork.k_plus_delta_k_polish.bottomRightCorner(numactive_constraints, numactive_constraints).diagonal().array() -= qpsettings.delta_osqp;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  qpwork.ldl.factorize(qpwork.k_plus_delta_k_polish.transpose(), stack);

  // Solve the first linear system K + Delta_K * hat_t = rhs, where rhs = [-g, b_low, l_low, b_up, u_up] and hat_t = [t, y_low, z_low, y_up, z_up]
  isize l_low_index = 0;
  isize u_up_index = 0;
  proxqp::dense::Vec<T> l_low(numactive_constraints_ineq_low);
  proxqp::dense::Vec<T> u_up(numactive_constraints_ineq_up);
  for (proxqp::isize i = 0; i < qpmodel.n_in; ++i) {
    if (qpwork.active_set_low_ineq(i)) {
      l_low(l_low_index) = qpwork.l_scaled(i);
      ++l_low_index;
    } 
    if (qpwork.active_set_up_ineq(i)) {
      u_up(u_up_index) = qpwork.u_scaled(i);
      ++u_up_index;
    }
  }

  isize b_low_index = 0;
  isize b_up_index = 0;
  proxqp::dense::Vec<T> b_low(numactive_constraints_eq_low);
  proxqp::dense::Vec<T> b_up(numactive_constraints_eq_up);
  for (proxqp::isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      b_low(b_low_index) = qpwork.b_scaled(i);
      ++b_low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      b_up(b_up_index) = qpwork.b_scaled(i);
      ++b_up_index;
    }
  }

  proxqp::dense::Vec<T> g_polish_rhs(qpmodel.dim + numactive_constraints);
  isize line = qpmodel.dim;
  g_polish_rhs.head(line) = - qpwork.g_scaled;
  // std::cout << "-g: " << - qpwork.g_scaled << std::endl;
  g_polish_rhs.segment(line, numactive_constraints_eq_low) = b_low;
  // std::cout << "b_low: " << b_low << std::endl;
  line+= numactive_constraints_eq_low;
  // l_low.setZero(); // Juste pour le test
  g_polish_rhs.segment(line, numactive_constraints_ineq_low) = l_low;
  // std::cout << "l_low: " << l_low << std::endl;
  line+= numactive_constraints_ineq_low;
  g_polish_rhs.segment(line, numactive_constraints_eq_up) = b_up;
  // std::cout << "b_up: " << b_up << std::endl;
  g_polish_rhs.tail(numactive_constraints_ineq_up) = u_up;
  // std::cout << "u_up: " << u_up << std::endl;

  isize inner_pb_dim = qpmodel.dim + numactive_constraints;
  proxqp::dense::Vec<T> hat_t = g_polish_rhs;

  // std::cout << "K + Delta_K: " << qpwork.k_plus_delta_k_polish << std::endl;
  // std::cout << "Max value in K + Delta_K: " << qpwork.k_plus_delta_k_polish.maxCoeff() << std::endl;
  // std::cout << "Min value in K + Delta_K: " << qpwork.k_plus_delta_k_polish.minCoeff() << std::endl;

  // std::cout << "g_polish_rhs: " << g_polish_rhs << std::endl;
  // std::cout << "Max value in g_polish_rhs: " << g_polish_rhs.maxCoeff() << std::endl;
  // std::cout << "Min value in g_polish_rhs: " << g_polish_rhs.minCoeff() << std::endl;

  proxqp::dense::solve_linear_system(hat_t,
                                     qpmodel,
                                     qpresults,
                                     qpwork,
                                     n_constraints,
                                     dense_backend,
                                     inner_pb_dim,
                                     stack);
  // std::cout << "hat_t: " << hat_t << std::endl;

  // Iterative refinement
  
  for (proxqp::i64 iter = 0; iter < qpsettings.nb_polish_iter; ++iter) {
    proxqp::dense::Vec<T> rhs_polish = g_polish_rhs - qpwork.k_polish * hat_t;
    proxqp::dense::Vec<T> delta_hat_t = rhs_polish;
    proxqp::dense::solve_linear_system(delta_hat_t,
                                       qpmodel,
                                       qpresults,
                                       qpwork,
                                       n_constraints,
                                       dense_backend,
                                       inner_pb_dim,
                                       stack);
    hat_t = hat_t + delta_hat_t;
  }
  // std::cout << "hat_t: " << hat_t << std::endl;

  // Update of the primal and dual variables

  qpresults.x = hat_t.head(qpmodel.dim);

  isize y_low_index = 0;
  isize z_low_index = 0;
  isize y_up_index = 0;
  isize z_up_index = 0;
  for (proxqp::isize i = 0; i < qpmodel.n_eq; ++i) {
    if (qpwork.active_set_low_eq(i)) {
      qpresults.y(i) = hat_t(qpmodel.dim + y_low_index); 
      ++y_low_index;
    }
    if (qpwork.active_set_up_eq(i)) {
      qpresults.y(i) = hat_t(qpmodel.dim + numactive_constraints_eq_low + numactive_constraints_ineq_low + y_up_index); 
      ++y_up_index;
    }
  }

  for (proxqp::isize i = 0; i < qpmodel.n_in; ++i) {
    if (qpwork.active_set_low_ineq(i)) {
      qpresults.z(i) = hat_t(qpmodel.dim + numactive_constraints_eq_low + z_low_index); 
      ++z_low_index;
    }
    if (qpwork.active_set_up_ineq(i)) {
      qpresults.z(i) = hat_t(qpmodel.dim + numactive_constraints_eq_low + numactive_constraints_ineq_low + numactive_constraints_eq_up + z_up_index); 
      ++z_up_index;
    }
  }

  // Prints
  // std::cout << "Variables after polishing: " << std::endl;
  // std::cout << "x: " << qpresults.x << std::endl;
  // std::cout << "y: " << qpresults.y << std::endl;
  // std::cout << "z: " << qpresults.z << std::endl;

}
// /*!
//  * Solves the KKT system in the case of equality constraints only.
//  *
//  * @param qpwork solver workspace.
//  * @param qpmodel QP problem model as defined by the user (without any scaling
//  * performed).
//  * @param qpsettings solver settings.
//  * @param qpresults solver results.
//  * @param ruiz ruiz preconditioner.
//  */
// template<typename T>
// bool
// solve_eq_constraints_system(const proxqp::Settings<T>& qpsettings,
//                             const proxqp::dense::Model<T>& qpmodel,
//                             proxqp::Results<T>& qpresults,
//                             proxqp::dense::Workspace<T>& qpwork,
//                             const bool box_constraints,
//                             const proxqp::isize n_constraints,
//                             proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
//                             const proxqp::DenseBackend dense_backend,
//                             const proxqp::HessianType hessian_type)
// {
//   using namespace proxsuite::proxqp;
//   // Solves the linear system
//   // ( (H + rho * I, A^T)  (A, -\mu_eq * I) ) ( x, y ) = ( rho * x_prev - g, b - \mu_eq * y_prev ) 
//   // Formulation in proxqp paper, equality case
//   qpwork.rhs.setZero();
//   qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
//   qpwork.rhs.tail(qpmodel.n_eq) = qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;
//   isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq;
//   proxsuite::linalg::veg::dynstack::DynStackMut stack{
//     proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
//   };
//   proxqp::dense::solve_linear_system(qpwork.rhs,
//                                      qpmodel,
//                                      qpresults,
//                                      qpwork,
//                                      n_constraints,
//                                      dense_backend,
//                                      inner_pb_dim,
//                                      stack);
//   /// New (x, y)
//   qpresults.x = qpwork.rhs.head(qpmodel.dim);
//   qpresults.y = qpwork.rhs.tail(qpmodel.n_eq);
//   std::cout << "x: " << qpresults.x << std::endl;
//   std::cout << "y: " << qpresults.y << std::endl;
//   // Delta x and y (and z equal to 0)
//   proxqp::dense::Vec<T> dx = qpresults.x - qpwork.x_prev; // TODO: Dirty way to define it ? (Compared to proxq code, line "auto dx = qpwork.dw_aug.head(qpmodel.dim);")
//   proxqp::dense::Vec<T> dy = qpresults.y - qpwork.y_prev;
//   proxqp::dense::Vec<T> dz = proxqp::dense::Vec<T>::Zero(qpmodel.n_in);
//   // New right hand side 
//   qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
//   qpwork.rhs.tail(qpmodel.n_eq) = qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;
//   // Prints x, y
//   std::cout << "x: " << qpresults.x << std::endl;
//   std::cout << "y: " << qpresults.y << std::endl;
//   // std::cout << "z: " << qpresults.z << std::endl;
//   // // Computing the resuduals: 
//   // // Note: The linear system in proxqp solves finds dx, dy and so on => I change the definitions
//   // // of Adx, etc (compared to proxqp solver)
//   auto& Hdx = qpwork.Hdx;
//   auto& Adx = qpwork.Adx;
//   auto& ATdy = qpwork.CTz;
//   switch (hessian_type) {
//     case HessianType::Zero:
//       break;
//     case HessianType::Dense:
//       Hdx.noalias() = qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * dx;
//       break;
//     case HessianType::Diagonal:
//   #ifndef NDEBUG
//       PROXSUITE_THROW_PRETTY(!qpwork.H_scaled.isDiagonal(),
//                               std::invalid_argument,
//                               "H is not diagonal.");
//   #endif
//       Hdx.array() = qpwork.H_scaled.diagonal().array() * dx.array();
//       break;
//   }
//   ATdy.noalias() = qpwork.A_scaled.transpose() * dy;
//   Adx.noalias() = qpwork.A_scaled * dx;
//   auto& Cdx = qpwork.Cdx;
//   LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
//   if (qpmodel.n_in > 0) {
//     Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
//     CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
//   }
//   // Check the feasibility (code inspired from the proxqp newton function)
//   bool is_primal_infeasible = proxsuite::proxqp::dense::global_primal_residual_infeasibility(
//     VectorViewMut<T>{ from_eigen, ATdy },
//     VectorViewMut<T>{ from_eigen, CTdz  },
//     VectorViewMut<T>{ from_eigen, dy },   
//     VectorViewMut<T>{ from_eigen, dz },
//     qpwork,
//     qpmodel,
//     qpsettings,
//     box_constraints,
//     ruiz);
//   std::cout << "is_primal_infeasible: " << is_primal_infeasible << std::endl;
//   bool is_dual_infeasible = proxsuite::proxqp::dense::global_dual_residual_infeasibility(
//     VectorViewMut<T>{ from_eigen, Adx },
//     VectorViewMut<T>{ from_eigen, Cdx }, 
//     VectorViewMut<T>{ from_eigen, Hdx },
//     VectorViewMut<T>{ from_eigen, dx },
//     qpwork,
//     qpsettings,
//     qpmodel,
//     box_constraints,
//     ruiz);
//   if (is_primal_infeasible) {
//     qpresults.info.status = QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE;
//   } else if (is_dual_infeasible) {
//     qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
//   }
//   if ((qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE &&
//         !qpsettings.primal_infeasibility_solving) ||
//       qpresults.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
//     // certificate of infeasibility
//     return true; // Instead of doing break; in one single big qp_solve function
//   } else {
//     return false;
//   }
//   // TODO: Keep this function while the inequality case is in progress
// }

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
                                    
  T duality_gap(0);                 // xHx + gTx + bTy + zTu + zTl
  T rhs_duality_gap(0);             // max(abs(each term))
  T scaled_eps(qpsettings.eps_abs); // eps_abs

  // Scaled variables used for the updates of mu_eq and mu_in
  T scaled_primal_feasibility_lhs(0);      // max(scaled_primal_feasibility_eq_lhs, scaled_primal_feasibility_in_lhs)
  T scaled_primal_feasibility_eq_rhs_0(0); // norm(scaled(Ax))
  T scaled_primal_feasibility_in_rhs_0(0); // norm(scaled(Cx))
  T scaled_primal_feasibility_eq_lhs(0);   // norm(scaled(Ax - b))
  T scaled_primal_feasibility_in_lhs(0);    // norm(scaled([Cx - u]+ + [Cx - l]-))

  T scaled_dual_feasibility_lhs(0);
  T scaled_dual_feasibility_rhs_0(0);
  T scaled_dual_feasibility_rhs_1(0);
  T scaled_dual_feasibility_rhs_3(0);

  setup_factorization_full_kkt(qpwork,
                               qpmodel,
                               qpresults,
                               dense_backend,
                               n_constraints);

  // std::cout << "KKT matrix: " << qpwork.kkt << std::endl;

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

    bool stopping_criteria_bis = admm(qpsettings,
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

  

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Specific: Update the proximal parameters mu_eq and mu_in

    if (qpsettings.update_mu_osqp){

      if (iter == 0){
        qpwork.timer_mu_update.stop();
        qpwork.timer_mu_update.start();       
        qpwork.time_since_last_mu_update = 0; 
      }

      qpwork.time_since_last_mu_update = qpwork.timer_mu_update.elapsed().user;
      bool time_condition = qpwork.time_since_last_mu_update > 
                            qpsettings.ratio_time_mu_update * qpwork.factorization_time_full_kkt;

      if (time_condition){
        
        T update_mu_criteria = compute_update_mu_criteria(qpwork,
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
        // std::cout << "update_mu_criteria: " << update_mu_criteria << std::endl;

        bool update_mu = update_mu_criteria < qpsettings.ratio_value_mu_update_inv ||
                        update_mu_criteria > qpsettings.ratio_value_mu_update;
        // std::cout << "Value condition: " << update_mu << std::endl;
        if (update_mu) {

          // Compute the new values of mu_eq and mu_in
          T new_mu_eq = qpresults.info.mu_eq / update_mu_criteria;
          T new_mu_in = qpresults.info.mu_in / update_mu_criteria; 
          T new_mu_eq_inv = qpresults.info.mu_eq_inv * update_mu_criteria; 
          T new_mu_in_inv = qpresults.info.mu_in_inv * update_mu_criteria; 

          // Update the parameters in the KKT matrix
          ++qpresults.info.mu_updates;
          proxsuite::proxqp::dense::mu_update(qpmodel,
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

    // Prints
    // std::cout << "Number of updates of mu : " << qpresults.info.mu_updates << std::endl;
    // std::cout << "mu_eq: " << qpresults.info.mu_eq << std::endl;
    // std::cout << "mu_in: " << qpresults.info.mu_in << std::endl;

    // TODO: For the moment, I do not code cold restart strategy for mu_eq and mu_in, 
    // as it is not in the OSQP paper, see later if needed
    }

  }

  // Polishing
  if (qpsettings.polish){
    if (qpresults.info.status == proxqp::QPSolverOutput::PROXQP_SOLVED) {

      polish(qpsettings,
         qpmodel, 
         qpresults, 
         qpwork, 
         box_constraints, 
         n_constraints,
         dense_backend, 
         hessian_type);

      proxqp::dense::global_primal_residual(qpmodel,
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

      proxqp::dense::global_dual_residual(qpresults,
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

  // std::cout << "Factorization time full kkt: " << qpwork.factorization_time_full_kkt << std::endl;

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

// TODO: Lighten the namespaces (due to common files)
// TODO: Eventually clean the code (like common functions names, or boolean as the result of a big function)

// TODO: In some cases (different qpmodel.dim values in osqp_overview-simple), the duality_gap does not converge to 0
// -> Adapt the definition for it (probably comes from the infinite bound values)