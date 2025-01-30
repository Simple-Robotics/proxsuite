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
  // We add the norm of zeta (z in osqp paper) in the max for discouting the primal residual

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
  
  // TODO: Also add the term norm(scaled(zeta)) in the max. To do it:
  // - First manage to scale and unscale it in the whole algorithm when needed
  // - Then compute it directly here
  // !! This is important, because I already coded the ADMM, but with unscaled zeta, which do not reflect the true algorithm

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

  T primal_term = scaled_primal_feasibility_lhs / std::max(scaled_primal_feasibility_eq_rhs_0,
                                                  std::max(proxqp::dense::infty_norm(qpwork.zeta_eq), 
                                                  std::max(proxqp::dense::infty_norm(qpwork.zeta_in), 
                                                           scaled_primal_feasibility_in_rhs_0)));

  T dual_term = scaled_dual_feasibility_lhs / std::max(scaled_dual_feasibility_rhs_0, 
                                              std::max(scaled_dual_feasibility_rhs_1, 
                                              std::max(proxqp::dense::infty_norm(qpwork.g_scaled),
                                                       scaled_dual_feasibility_rhs_3)));

  T update_mu_criteria = std::sqrt(primal_term / dual_term);

  // TODO: See how to manage the case where the dual term is very little (add some epsilon somewhere ?)

  return update_mu_criteria;

}
/*!
 * Minimum vector element-wise.
 *
 * @param vec1 first vector.
 * @param vec2 second vector.
 */
template<typename T>
proxqp::dense::Vec<T> minVec(const proxqp::dense::Vec<T>& vec1, const proxqp::dense::Vec<T>& vec2) {
    assert(vec1.size() == vec2.size() && "Vectors must be of the same size");
    return vec1.cwiseMin(vec2);
}
/*!
 * maximum vector element-wise.
 *
 * @param vec1 first vector.
 * @param vec2 second vector.
 */
template<typename T>
proxqp::dense::Vec<T> maxVec(const proxqp::dense::Vec<T>& vec1, const proxqp::dense::Vec<T>& vec2) {
    assert(vec1.size() == vec2.size() && "Vectors must be of the same size");
    return vec1.cwiseMax(vec2);
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

  isize n_c_f = qpwork.n_c;

  // suppression pour le nouvel active set, ajout dans le nouvel unactive set
  // [Not done here, because we build the complete KKT matrix once before the ADMM iterations]
  
  // ajout au nouvel active set, suppression pour le nouvel unactive set

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
            col.tail(n_eq + n_c_f).setZero();
            col[n + n_eq + n_c + k] = mu_in_neg;
          }
          qpwork.timer_full_fact.start();
          qpwork.ldl.insert_block_at(n + n_eq + n_c, new_cols, stack);
          qpwork.timer_full_fact.stop();
          qpwork.factorization_time_full_kkt = qpwork.timer_full_fact.elapsed().user;
        } break;
        case DenseBackend::PrimalLDLT: { // TODO: Code this later, after the PrimalDualLDLT case
        } break;
        case DenseBackend::Automatic: {
        } break;
      }
    }
  }

  qpwork.n_c = n_c_f;

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

  // std::cout << "rhs: " << qpwork.rhs << std::endl;
  
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
  // std::cout << "x_tilde: " << x_tilde << std::endl;
  proxqp::dense::Vec<T> nu_eq = qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq);
  // std::cout << "nu_eq: " << nu_eq << std::endl;
  proxqp::dense::Vec<T> nu_in = qpwork.rhs.tail(qpmodel.n_in);
  // std::cout << "nu_in: " << nu_in << std::endl;

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
  proxqp::dense::Vec<T> new_zeta_in = 
    maxVec(qpwork.l_scaled, 
    minVec(sum_zeta_in_z,
                qpwork.u_scaled));

  // Update of y and z
  qpresults.y = qpresults.y + qpresults.info.mu_eq_inv * (qpsettings.alpha_osqp * zeta_tilde_eq + (1 - qpsettings.alpha_osqp) * qpwork.zeta_eq - new_zeta_eq);
  qpresults.z = qpresults.z + qpresults.info.mu_in_inv * (qpsettings.alpha_osqp * zeta_tilde_in + (1 - qpsettings.alpha_osqp) * qpwork.zeta_in - new_zeta_in);

  // Update of the zeta's
  qpwork.zeta_eq = new_zeta_eq;
  qpwork.zeta_in = new_zeta_in;

  // Delta x, y and z
  proxqp::dense::Vec<T> dx = qpresults.x - qpwork.x_prev;
  proxqp::dense::Vec<T> dy = qpresults.y - qpwork.y_prev;
  proxqp::dense::Vec<T> dz = qpresults.z - qpwork.z_prev;

  // New right hand side
  qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.zeta_eq - qpresults.info.mu_eq * qpresults.y;
  qpwork.rhs.tail(qpmodel.n_in) = qpwork.zeta_in - qpresults.info.mu_in * qpresults.z;

  // Prints x, y, z
  std::cout << "x: " << qpresults.x << std::endl;
  std::cout << "y: " << qpresults.y << std::endl;
  std::cout << "z: " << qpresults.z << std::endl;

  // End of corpus of the ADMM iteration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


  // Computing the residuals
  // TODO: Put in common and / or delete the equality case version, because this is a pure copy paste

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

  // Check the feasibility (code inspired from the proxqp newton function)
  // TODO: In proxqp solver, there is a condition to test this which is "qpsettings.primal_infeasibility_solving == true"
  // Understand why it is used, since it would make me fail tests here (if set to false)

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
    // certificate of infeasibility
    return true; // Instead of doing break; in one single big qp_solve function
  } else {
    return false;
  }

  // TODO: Once the inequality case is coded, 
  // Handle (here) the closest feasible primal solution in case of primal infeasibility

}
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

  // Solves the linear system
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
  std::cout << "x: " << qpresults.x << std::endl;
  std::cout << "y: " << qpresults.y << std::endl;

  // Delta x and y (and z equal to 0)
  proxqp::dense::Vec<T> dx = qpresults.x - qpwork.x_prev; // TODO: Dirty way to define it ? (Compared to proxq code, line "auto dx = qpwork.dw_aug.head(qpmodel.dim);")
  proxqp::dense::Vec<T> dy = qpresults.y - qpwork.y_prev;
  proxqp::dense::Vec<T> dz = proxqp::dense::Vec<T>::Zero(qpmodel.n_in);

  // New right hand side 
  qpwork.rhs.head(qpmodel.dim) = qpresults.info.rho * qpresults.x - qpwork.g_scaled;
  qpwork.rhs.tail(qpmodel.n_eq) = qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y;

  // Prints x, y
  std::cout << "x: " << qpresults.x << std::endl;
  std::cout << "y: " << qpresults.y << std::endl;
  // std::cout << "z: " << qpresults.z << std::endl;

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

  auto& Cdx = qpwork.Cdx;
  LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
  if (qpmodel.n_in > 0) {
    Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
    CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
  }

  // Check the feasibility (code inspired from the proxqp newton function)
  // TODO: In proxqp solver, there is a condition to test this which is "qpsettings.primal_infeasibility_solving == true"
  // Understand why it is used, since it would make me fail tests here (if set to false)

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
  std::cout << "is_primal_infeasible: " << is_primal_infeasible << std::endl;

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
    // certificate of infeasibility
    return true; // Instead of doing break; in one single big qp_solve function
  } else {
    return false;
  }

  // TODO: Once the inequality case is coded, 
  // Handle (here) the closest feasible primal solution in case of primal infeasibility

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

  T scaled_primal_feasibility_lhs(0);      // max(scaled_primal_feasibility_eq_lhs, scaled_primal_feasibility_in_lhs)
  T scaled_primal_feasibility_eq_rhs_0(0); // norm(scaled(Ax))
  T scaled_primal_feasibility_in_rhs_0(0); // norm(scaled(Cx))
  T scaled_primal_feasibility_eq_lhs(0);   // norm(scaled(Ax - b))
  T scaled_primal_feasibility_in_lhs(0);    // norm(scaled([Cx - u]+ + [Cx - l]-))

  T scaled_dual_feasibility_lhs(0);
  T scaled_dual_feasibility_rhs_0(0);
  T scaled_dual_feasibility_rhs_1(0);
  T scaled_dual_feasibility_rhs_3(0);

  // Code for the inequality case (general case of OSQP)
  // Here I define and factorize the new KKT matrix (we add the inequality constraints within the matrix C)
  // The idea is to do it before the loop, which is different than the proxqp implementation, because the KKT matrix is 
  // constant in its indexes (then not updated in the loop)

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

    // bool stopping_criteria_bis = solve_eq_constraints_system(qpsettings, // TODO: Is it well written ? (both solve and boolean or stopping)
    //                             qpmodel, 
    //                             qpresults, 
    //                             qpwork, 
    //                             box_constraints, 
    //                             n_constraints,
    //                             ruiz, 
    //                             dense_backend, 
    //                             hessian_type);

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

      // if (iter == 25){
      //   break;
      // }

      if (time_condition){
        
        // T update_mu_criteria = 1; // Generaic value to test the pipeline
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
        std::cout << "update_mu_criteria: " << update_mu_criteria << std::endl;

        bool update_mu = update_mu_criteria < qpsettings.ratio_value_mu_update_inv ||
                        update_mu_criteria > qpsettings.ratio_value_mu_update;
        std::cout << "Value condition: " << update_mu << std::endl;
        if (update_mu) {

          // Compute the new values of mu_eq and mu_in
          T new_mu_eq = qpresults.info.mu_eq / update_mu_criteria; // Because mu = inverse of rho in osqp paper
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

  // End

  std::cout << "Factorization time full kkt: " << qpwork.factorization_time_full_kkt << std::endl;

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


