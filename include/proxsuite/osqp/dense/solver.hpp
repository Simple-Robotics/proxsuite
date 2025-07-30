//
// Copyright (c) 2025 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_OSQP_DENSE_SOLVER_HPP
#define PROXSUITE_OSQP_DENSE_SOLVER_HPP

#include "proxsuite/proxqp/dense/preconditioner/ruiz.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/dense/helpers.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include "proxsuite/proxqp/dense/solver.hpp"
#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/osqp/dense/utils.hpp"
#include <iostream>
#include <iomanip>

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
    qpwork.b_scaled - qpresults.info.mu_eq * qpresults.y; // zeta_eq = b
  qpwork.rhs.tail(n_constraints) =
    qpresults.zeta_in - qpresults.info.mu_in * qpresults.z;

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
  qpwork.x_tilde = qpwork.rhs.head(qpmodel.dim);
  qpwork.nu_eq = qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq);
  qpwork.nu_in = qpwork.rhs.tail(n_constraints);

  // Update the variables
  qpwork.zeta_tilde_eq =
    qpwork.b_scaled +
    qpresults.info.mu_eq * (qpwork.nu_eq - qpresults.y); // zeta_eq = b
  qpwork.zeta_tilde_in =
    qpresults.zeta_in + qpresults.info.mu_in * (qpwork.nu_in - qpresults.z);

  qpresults.x = qpsettings.alpha_osqp * qpwork.x_tilde +
                (1 - qpsettings.alpha_osqp) * qpresults.x;

  qpresults.zeta_eq = qpwork.b_scaled; // zeta_eq = b
  qpwork.zeta_in_next = qpsettings.alpha_osqp * qpwork.zeta_tilde_in +
                        (1 - qpsettings.alpha_osqp) * qpresults.zeta_in +
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

  qpresults.y =
    qpresults.y +
    qpresults.info.mu_eq_inv *
      (qpsettings.alpha_osqp * qpwork.zeta_tilde_eq +
       (1 - qpsettings.alpha_osqp) * qpresults.zeta_eq - qpresults.zeta_eq);
  qpresults.z =
    qpresults.z +
    qpresults.info.mu_in_inv *
      (qpsettings.alpha_osqp * qpwork.zeta_tilde_in +
       (1 - qpsettings.alpha_osqp) * qpresults.zeta_in - qpwork.zeta_in_next);

  qpresults.zeta_in = qpwork.zeta_in_next;
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
  preconditioner::RuizEquilibration<T>& ruiz)
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

  //////////////////////////////////////////////////////////////////////////////////////////

  if (qpsettings.verbose) {
    proxsuite::osqp::dense::print_setup_header(qpsettings,
                                               qpresults,
                                               qpmodel,
                                               box_constraints,
                                               dense_backend,
                                               hessian_type);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  if (qpwork.dirty) { // the following is used when a solve has already been
                      // executed (and without any intermediary model update)
    switch (qpsettings.initial_guess) {
      case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
        qpwork.cleanup(box_constraints);
        qpresults.cleanup(qpsettings);
        break;
      }
      case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
        // keep solutions but restart workspace and results
        qpwork.cleanup(box_constraints);
        qpresults.cold_start(qpsettings);
        ruiz.scale_primal_in_place(
          { proxsuite::proxqp::from_eigen, qpresults.x });
        ruiz.scale_dual_in_place_eq(
          { proxsuite::proxqp::from_eigen, qpresults.y });
        ruiz.scale_dual_in_place_in(
          { proxsuite::proxqp::from_eigen, qpresults.z.head(qpmodel.n_in) });
        if (box_constraints) {
          ruiz.scale_box_dual_in_place_in(
            { proxsuite::proxqp::from_eigen, qpresults.z.tail(qpmodel.dim) });
        }
        break;
      }
      case InitialGuessStatus::NO_INITIAL_GUESS: {
        qpwork.cleanup(box_constraints);
        qpresults.cleanup(qpsettings);
        break;
      }
      case InitialGuessStatus::WARM_START: {
        qpwork.cleanup(box_constraints);
        qpresults.cold_start(
          qpsettings); // because there was already a solve,
                       // precond was already computed if set so
        ruiz.scale_primal_in_place(
          { proxsuite::proxqp::from_eigen,
            qpresults
              .x }); // it contains the value given in entry for warm start
        ruiz.scale_dual_in_place_eq(
          { proxsuite::proxqp::from_eigen, qpresults.y });
        ruiz.scale_dual_in_place_in(
          { proxsuite::proxqp::from_eigen, qpresults.z.head(qpmodel.n_in) });
        if (box_constraints) {
          ruiz.scale_box_dual_in_place_in(
            { proxsuite::proxqp::from_eigen, qpresults.z.tail(qpmodel.dim) });
        }
        break;
      }
      case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
        // keep workspace and results solutions except statistics
        // std::cout << "i keep previous solution" << std::endl;
        qpresults.cleanup_statistics();
        ruiz.scale_primal_in_place(
          { proxsuite::proxqp::from_eigen, qpresults.x });
        ruiz.scale_dual_in_place_eq(
          { proxsuite::proxqp::from_eigen, qpresults.y });
        ruiz.scale_dual_in_place_in(
          { proxsuite::proxqp::from_eigen, qpresults.z.head(qpmodel.n_in) });
        if (box_constraints) {
          ruiz.scale_box_dual_in_place_in(
            { proxsuite::proxqp::from_eigen, qpresults.z.tail(qpmodel.dim) });
        }
        break;
      }
    }
    if (qpsettings.initial_guess !=
        InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT) {
      switch (hessian_type) {
        case HessianType::Zero:
          break;
        case HessianType::Dense:
          qpwork.H_scaled = qpmodel.H;
          break;
        case HessianType::Diagonal:
          qpwork.H_scaled = qpmodel.H;
          break;
      }
      qpwork.g_scaled = qpmodel.g;
      qpwork.A_scaled = qpmodel.A;
      qpwork.b_scaled = qpmodel.b;
      qpwork.C_scaled = qpmodel.C;
      qpwork.u_scaled = qpmodel.u;
      qpwork.l_scaled = qpmodel.l;
      proxsuite::proxqp::dense::setup_equilibration(
        qpwork,
        qpsettings,
        box_constraints,
        hessian_type,
        ruiz,
        false); // reuse previous equilibration
      proxsuite::proxqp::dense::setup_factorization(
        qpwork, qpmodel, qpresults, dense_backend, hessian_type);
    }
    if (qpsettings.initial_guess ==
        InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS) {
      compute_equality_constrained_initial_guess(qpwork,
                                                 qpsettings,
                                                 qpmodel,
                                                 n_constraints,
                                                 dense_backend,
                                                 hessian_type,
                                                 qpresults);
    }
    setup_factorization_complete_kkt(
      qpresults, qpmodel, qpwork, n_constraints, dense_backend);
  } else { // the following is used for a first solve after initializing or
           // updating the Qp object
    switch (qpsettings.initial_guess) {
      case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
        proxsuite::proxqp::dense::setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        compute_equality_constrained_initial_guess(qpwork,
                                                   qpsettings,
                                                   qpmodel,
                                                   n_constraints,
                                                   dense_backend,
                                                   hessian_type,
                                                   qpresults);
        setup_factorization_complete_kkt(
          qpresults, qpmodel, qpwork, n_constraints, dense_backend);
        break;
      }
      case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
        //!\ TODO in a quicker way
        ruiz.scale_primal_in_place(
          { proxsuite::proxqp::from_eigen,
            qpresults
              .x }); // meaningful for when there is an upate of the model and
                     // one wants to warm start with previous result
        ruiz.scale_dual_in_place_eq(
          { proxsuite::proxqp::from_eigen, qpresults.y });
        ruiz.scale_dual_in_place_in(
          { proxsuite::proxqp::from_eigen, qpresults.z.head(qpmodel.n_in) });
        if (box_constraints) {
          ruiz.scale_box_dual_in_place_in(
            { proxsuite::proxqp::from_eigen, qpresults.z.tail(qpmodel.dim) });
        }
        setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        setup_factorization_complete_kkt(
          qpresults, qpmodel, qpwork, n_constraints, dense_backend);
        break;
      }
      case InitialGuessStatus::NO_INITIAL_GUESS: {
        setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        setup_factorization_complete_kkt(
          qpresults, qpmodel, qpwork, n_constraints, dense_backend);
        break;
      }
      case InitialGuessStatus::WARM_START: {
        //!\ TODO in a quicker way
        ruiz.scale_primal_in_place(
          { proxsuite::proxqp::from_eigen, qpresults.x });
        ruiz.scale_dual_in_place_eq(
          { proxsuite::proxqp::from_eigen, qpresults.y });
        ruiz.scale_dual_in_place_in(
          { proxsuite::proxqp::from_eigen, qpresults.z.head(qpmodel.n_in) });
        if (box_constraints) {
          ruiz.scale_box_dual_in_place_in(
            { proxsuite::proxqp::from_eigen, qpresults.z.tail(qpmodel.dim) });
        }
        setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        setup_factorization_complete_kkt(
          qpresults, qpmodel, qpwork, n_constraints, dense_backend);
        break;
      }
      case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
        // std::cout << "i refactorize from previous solution" << std::endl;
        ruiz.scale_primal_in_place(
          { proxsuite::proxqp::from_eigen,
            qpresults
              .x }); // meaningful for when there is an upate of the model and
                     // one wants to warm start with previous result
        ruiz.scale_dual_in_place_eq(
          { proxsuite::proxqp::from_eigen, qpresults.y });
        ruiz.scale_dual_in_place_in(
          { proxsuite::proxqp::from_eigen, qpresults.z.head(qpmodel.n_in) });
        if (box_constraints) {
          ruiz.scale_box_dual_in_place_in(
            { proxsuite::proxqp::from_eigen, qpresults.z.tail(qpmodel.dim) });
        }
        if (qpwork.refactorize) { // refactorization only when one of the
                                  // matrices has changed or one proximal
                                  // parameter has changed
          setup_factorization(
            qpwork, qpmodel, qpresults, dense_backend, hessian_type);
          setup_factorization_complete_kkt(
            qpresults, qpmodel, qpwork, n_constraints, dense_backend);
          break;
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

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

  //////////////////////////////////////////////////////////////////////////////////////////

  for (i64 iter = 0; iter < qpsettings.max_iter; ++iter) {

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

    T rhs_pri(scaled_eps);
    if (qpsettings.eps_rel != 0) {
      rhs_pri += qpsettings.eps_rel * std::max(primal_feasibility_eq_rhs_0,
                                               primal_feasibility_in_rhs_0);
    }
    bool is_primal_feasible = primal_feasibility_lhs <= rhs_pri;

    T rhs_dua(qpsettings.eps_abs);
    if (qpsettings.eps_rel != 0) {
      rhs_dua +=
        qpsettings.eps_rel *
        std::max(
          std::max(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
          std::max(dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2));
    }

    bool is_dual_feasible = dual_feasibility_lhs <= rhs_dua;

    //////////////////////////////////////////////////////////////////////////////////////////

    if (qpsettings.verbose) {

      ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
      ruiz.unscale_dual_in_place_eq(
        VectorViewMut<T>{ from_eigen, qpresults.y });
      ruiz.unscale_dual_in_place_in(
        VectorViewMut<T>{ from_eigen, qpresults.z.head(qpmodel.n_in) });
      if (box_constraints) {
        ruiz.unscale_box_dual_in_place_in(
          VectorViewMut<T>{ from_eigen, qpresults.z.tail(qpmodel.dim) });
      }
      {
        qpresults.info.objValue = 0;
        for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
          qpresults.info.objValue +=
            0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
          qpresults.info.objValue +=
            qpresults.x(j) * T(qpmodel.H.col(j)
                                 .tail(qpmodel.dim - j - 1)
                                 .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
        }
        qpresults.info.objValue += (qpmodel.g).dot(qpresults.x);
      }
      std::cout << "\033[1;32m[iteration " << iter + 1 << "]\033[0m"
                << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | mu_eq=" << qpresults.info.mu_eq
                << " | mu_in=" << qpresults.info.mu_in << std::endl;
      ruiz.scale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
      ruiz.scale_dual_in_place_eq(VectorViewMut<T>{ from_eigen, qpresults.y });
      ruiz.scale_dual_in_place_in(
        VectorViewMut<T>{ from_eigen, qpresults.z.head(qpmodel.n_in) });
      if (box_constraints) {
        ruiz.scale_box_dual_in_place_in(
          VectorViewMut<T>{ from_eigen, qpresults.z.tail(qpmodel.dim) });
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////

    if (is_primal_feasible && is_dual_feasible) {
      if (qpsettings.check_duality_gap) {
        if (std::fabs(qpresults.info.duality_gap) <=
            qpsettings.eps_duality_gap_abs +
              qpsettings.eps_duality_gap_rel * rhs_duality_gap) {
          if (qpsettings.primal_infeasibility_solving &&
              qpresults.info.status ==
                QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
            qpresults.info.status =
              QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
          } else {
            qpresults.info.status = QPSolverOutput::PROXQP_SOLVED;
          }
          break;
        }
      } else {
        qpresults.info.status = QPSolverOutput::PROXQP_SOLVED;
        break;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////

    qpresults.info.iter_ext += 1; // We start a new external loop update

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    //////////////////////////////////////////////////////////////////////////////////////////

    admm_step(qpsettings,
              qpmodel,
              qpresults,
              qpwork,
              box_constraints,
              n_constraints,
              dense_backend);

    //////////////////////////////////////////////////////////////////////////////////////////

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
      qpwork.active_part_z.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
      CTdz.noalias() += qpwork.active_part_z.tail(qpmodel.dim);

      Cdx.tail(qpmodel.dim) = dx;
      Cdx.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
    }

    if (iter % qpsettings.frequence_infeasibility_check == 0 ||
        qpsettings.primal_infeasibility_solving) {
      // compute primal and dual infeasibility criteria
      bool is_primal_infeasible = global_primal_residual_infeasibility(
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
        break;
      } else if (is_dual_infeasible) {
        qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
        break;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////

    if ((qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE &&
         !qpsettings.primal_infeasibility_solving) ||
        qpresults.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      // certificate of infeasibility
      qpresults.x = qpwork.dw_aug.head(qpmodel.dim);
      qpresults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
      qpresults.z = qpwork.dw_aug.tail(n_constraints);
      break;
    }

    //////////////////////////////////////////////////////////////////////////////////////////

    T primal_feasibility_lhs_new(primal_feasibility_lhs);
    global_primal_residual(qpmodel,
                           qpresults,
                           qpsettings,
                           qpwork,
                           ruiz,
                           box_constraints,
                           primal_feasibility_lhs_new,
                           primal_feasibility_eq_rhs_0,
                           primal_feasibility_in_rhs_0,
                           primal_feasibility_eq_lhs,
                           primal_feasibility_in_lhs);

    is_primal_feasible =
      primal_feasibility_lhs_new <=
      (scaled_eps + qpsettings.eps_rel * std::max(primal_feasibility_eq_rhs_0,
                                                  primal_feasibility_in_rhs_0));
    qpresults.info.pri_res = primal_feasibility_lhs_new;
    if (is_primal_feasible) {
      T dual_feasibility_lhs_new(dual_feasibility_lhs);

      global_dual_residual(qpresults,
                           qpwork,
                           qpmodel,
                           box_constraints,
                           ruiz,
                           dual_feasibility_lhs_new,
                           dual_feasibility_rhs_0,
                           dual_feasibility_rhs_1,
                           dual_feasibility_rhs_3,
                           rhs_duality_gap,
                           duality_gap,
                           hessian_type);
      qpresults.info.dua_res = dual_feasibility_lhs_new;
      qpresults.info.duality_gap = duality_gap;

      is_dual_feasible =
        dual_feasibility_lhs_new <=
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
            if (qpsettings.primal_infeasibility_solving &&
                qpresults.info.status ==
                  QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
              qpresults.info.status =
                QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
            } else {
              qpresults.info.status = QPSolverOutput::PROXQP_SOLVED;
            }
          }
        } else {
          if (qpsettings.primal_infeasibility_solving &&
              qpresults.info.status ==
                QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
            qpresults.info.status =
              QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
          } else {
            qpresults.info.status = QPSolverOutput::PROXQP_SOLVED;
          }
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
  ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{ from_eigen, qpresults.y });
  ruiz.unscale_dual_in_place_in(
    VectorViewMut<T>{ from_eigen, qpresults.z.head(qpmodel.n_in) });
  if (box_constraints) {
    ruiz.unscale_box_dual_in_place_in(
      VectorViewMut<T>{ from_eigen, qpresults.z.tail(qpmodel.dim) });
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  {
    qpresults.info.objValue = 0;
    for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
      qpresults.info.objValue +=
        0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
      qpresults.info.objValue +=
        qpresults.x(j) * T(qpmodel.H.col(j)
                             .tail(qpmodel.dim - j - 1)
                             .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
    }
    qpresults.info.objValue += (qpmodel.g).dot(qpresults.x);
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  if (qpsettings.compute_timings) {
    qpresults.info.solve_time = qpwork.timer.elapsed().user; // in microseconds
    qpresults.info.run_time =
      qpresults.info.solve_time + qpresults.info.setup_time;
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  if (qpsettings.verbose) {
    std::cout << "-------------------SOLVER STATISTICS-------------------"
              << std::endl;
    std::cout << "total iter:     " << qpresults.info.iter << std::endl;
    std::cout << "mu updates:     " << qpresults.info.mu_updates << std::endl;
    std::cout << "objective:      " << qpresults.info.objValue << std::endl;
    switch (qpresults.info.status) {
      case QPSolverOutput::PROXQP_SOLVED: {
        std::cout << "status:         "
                  << "Solved" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_MAX_ITER_REACHED: {
        std::cout << "status:         "
                  << "Maximum number of iterations reached" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE: {
        std::cout << "status:         "
                  << "Primal infeasible" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_DUAL_INFEASIBLE: {
        std::cout << "status:         "
                  << "Dual infeasible" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: {
        std::cout << "status:         "
                  << "Solved closest primal feasible" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_NOT_RUN: {
        std::cout << "status:         "
                  << "Solver not run" << std::endl;
        break;
      }
    }

    if (qpsettings.compute_timings)
      std::cout << "run time [μs]:  " << qpresults.info.solve_time << std::endl;
    std::cout << "--------------------------------------------------------"
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  qpwork.dirty = true;
  qpwork.is_initialized = true;

  assert(!std::isnan(qpresults.info.pri_res));
  assert(!std::isnan(qpresults.info.dua_res));
  assert(!std::isnan(qpresults.info.duality_gap));

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_SOLVER_HPP */
