//
// Copyright (c) 2022 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_PROXQP_DENSE_SOLVER_HPP
#define PROXSUITE_PROXQP_DENSE_SOLVER_HPP

#include "proxsuite/fwd.hpp"
#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/dense/linesearch.hpp"
#include "proxsuite/proxqp/dense/helpers.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include <cmath>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>
#include <proxsuite/linalg/dense/ldlt.hpp>
#include <chrono>
#include <iomanip>

namespace proxsuite {
namespace proxqp {
namespace dense {

/*!
 * Performs a refactorization of the KKT matrix used by the solver.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param rho_new new primal proximal parameter used for the refactorization.
 */
template<typename T>
void
refactorize(const Model<T>& qpmodel,
            Results<T>& qpresults,
            Workspace<T>& qpwork,
            const isize n_constraints,
            const DenseBackend& dense_backend,
            T rho_new)
{

  if (!qpwork.constraints_changed && rho_new == qpresults.info.rho) {
    return;
  }

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT: {
      qpwork.kkt.diagonal().head(qpmodel.dim).array() +=
        rho_new - qpresults.info.rho;
      qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
        -qpresults.info.mu_eq;
      qpwork.ldl.factorize(qpwork.kkt.transpose(), stack);

      isize n = qpmodel.dim;
      isize n_eq = qpmodel.n_eq;
      isize n_in = qpmodel.n_in;
      isize n_c = qpwork.n_c;

      LDLT_TEMP_MAT(T, new_cols, n + n_eq + n_c, n_c, stack);
      T mu_in_neg(-qpresults.info.mu_in);
      for (isize i = 0; i < n_constraints; ++i) {
        isize j = qpwork.current_bijection_map[i];
        if (j < n_c) {
          auto col = new_cols.col(j);
          if (i >= n_in) {
            // I_scaled = D which is the diagonal matrix
            // scaling x
            // col(i-n_in) = ruiz.delta[i-qpmodel.n_in];
            col(i - n_in) = qpwork.i_scaled[i - qpmodel.n_in];
          } else {
            col.head(n) = qpwork.C_scaled.row(i);
          }
          col.segment(n, n_eq + n_c).setZero();
          col(n + n_eq + j) = mu_in_neg;
        }
      }
      qpwork.ldl.insert_block_at(n + n_eq, new_cols, stack);
    } break;
    case DenseBackend::PrimalLDLT: {
      qpwork.kkt.noalias() =
        qpwork.H_scaled + (qpwork.A_scaled.transpose() * qpwork.A_scaled) *
                            qpresults.info.mu_eq_inv;
      qpwork.kkt.diagonal().array() += qpresults.info.rho;
      for (isize i = 0; i < n_constraints; i++) {
        if (qpwork.active_inequalities(i)) {
          if (i >= qpmodel.n_in) {
            // box constraints
            qpwork.kkt(i - qpmodel.n_in, i - qpmodel.n_in) +=
              std::pow(qpwork.i_scaled(i - qpmodel.n_in), 2) *
              qpresults.info.mu_in_inv;
          } else {
            // generic ineq constraint
            qpwork.kkt.noalias() += qpwork.C_scaled.row(i).transpose() *
                                    qpwork.C_scaled.row(i) *
                                    qpresults.info.mu_in_inv;
          }
        }
      }
      qpwork.ldl.factorize(qpwork.kkt.transpose(), stack);
    } break;
    case DenseBackend::Automatic:
      break;
  }

  qpwork.constraints_changed = false;
}

/*!
 * Updates the dual proximal parameters of the solver (i.e., penalization
 * parameters of the primal-dual merit function).
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param mu_eq_new new dual equality constrained proximal parameter.
 * @param mu_in_new new dual inequality constrained proximal parameter.
 */
template<typename T>
void
mu_update(const Model<T>& qpmodel,
          Results<T>& qpresults,
          Workspace<T>& qpwork,
          isize n_constraints,
          const DenseBackend& dense_backend,
          T mu_eq_new,
          T mu_in_new)
{
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

  isize n = qpmodel.dim;
  isize n_eq = qpmodel.n_eq;
  isize n_c = qpwork.n_c;

  if ((n_eq + n_c) == 0) {
    return;
  }
  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT: {
      LDLT_TEMP_VEC_UNINIT(T, rank_update_alpha, n_eq + n_c, stack);

      rank_update_alpha.head(n_eq).setConstant(qpresults.info.mu_eq -
                                               mu_eq_new);
      rank_update_alpha.tail(n_c).setConstant(qpresults.info.mu_in - mu_in_new);

      {
        auto _indices = stack.make_new_for_overwrite(
          proxsuite::linalg::veg::Tag<isize>{}, n_eq + n_c);
        isize* indices = _indices.ptr_mut();
        for (isize k = 0; k < n_eq; ++k) {
          indices[k] = n + k;
        }
        for (isize k = 0; k < n_c; ++k) {
          indices[n_eq + k] = n + n_eq + k;
        }
        qpwork.ldl.diagonal_update_clobber_indices(
          indices, n_eq + n_c, rank_update_alpha, stack);
      }
    } break;
    case DenseBackend::PrimalLDLT: {
      // we refactorize there for the moment
      proxsuite::linalg::veg::dynstack::DynStackMut stack{
        proxsuite::linalg::veg::from_slice_mut,
        qpwork.ldl_stack.as_mut(),
      };
      // qpwork.kkt.noalias() = qpwork.H_scaled + (qpwork.A_scaled.transpose() *
      // qpwork.A_scaled) / mu_eq_new; qpwork.kkt.diagonal().array() +=
      // qpresults.info.rho; for (isize i = 0; i < n_constraints; i++){
      //   if (qpwork.active_inequalities(i)){
      //     if (i >=qpmodel.n_in){
      //       // box constraints
      //       qpwork.kkt(i-qpmodel.n_in,i-qpmodel.n_in) +=
      //       std::pow(qpwork.i_scaled(i-qpmodel.n_in),2) / mu_in_new ;
      //     } else{
      //       // generic ineq constraint
      //       qpwork.kkt.noalias() += qpwork.C_scaled.row(i).transpose() *
      //       qpwork.C_scaled.row(i) / mu_in_new ;
      //     }
      //   }
      // }
      // qpwork.ldl.factorize(qpwork.kkt.transpose(), stack);

      // mu update for C_J
      {
        LDLT_TEMP_MAT_UNINIT(T, new_cols, qpmodel.dim, qpwork.n_c, stack);
        qpwork.dw_aug.head(qpmodel.dim).setOnes();
        T delta_mu(mu_in_new - qpresults.info.mu_in_inv);
        qpwork.dw_aug.head(qpmodel.dim).array() *= delta_mu;
        for (isize i = 0; i < n_constraints; ++i) {
          isize j = qpwork.current_bijection_map[i];
          if (j < n_c) {
            auto col = new_cols.col(j);
            if (i >= qpmodel.n_in) {
              // box constraint
              col.setZero();
              col[i - qpmodel.n_in] = qpwork.i_scaled[i - qpmodel.n_in];
            } else {
              // generic ineq constraints
              col = qpwork.C_scaled.row(i);
            }
          }
        }
        qpwork.ldl.rank_r_update(
          new_cols, qpwork.dw_aug.head(qpmodel.dim), stack);
      }
      // mu update for A
      {
        LDLT_TEMP_MAT_UNINIT(T, new_cols, qpmodel.dim, qpmodel.n_eq, stack);
        new_cols = qpwork.A_scaled.transpose();
        qpwork.ldl.rank_r_update(
          new_cols, qpwork.dw_aug.head(qpmodel.dim), stack);
      }
    } break;
    case DenseBackend::Automatic:
      break;
  }
  qpwork.constraints_changed = true;
}
/*!
 * Derives the residual of the iterative refinement algorithm used for solving
 * associated linear systems of PROXQP algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param inner_pb_dim dimension of the linear system.
 */
template<typename T>
void
iterative_residual(const Model<T>& qpmodel,
                   Results<T>& qpresults,
                   Workspace<T>& qpwork,
                   const isize n_constraints,
                   isize inner_pb_dim,
                   const HessianType& hessian_type)
{
  auto& Hdx = qpwork.Hdx;
  auto& Adx = qpwork.Adx;
  auto& ATdy = qpwork.CTz;
  qpwork.err.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
  switch (hessian_type) {
    case HessianType::Zero:
      break;
    case HessianType::Dense:
      Hdx.noalias() = qpwork.H_scaled.template selfadjointView<Eigen::Lower>() *
                      qpwork.dw_aug.head(qpmodel.dim);
      qpwork.err.head(qpmodel.dim).noalias() -= Hdx;
      break;
    case HessianType::Diagonal:
#ifndef NDEBUG
      PROXSUITE_THROW_PRETTY(!qpwork.H_scaled.isDiagonal(),
                             std::invalid_argument,
                             "H is not diagonal.");
#endif
      Hdx.array() = qpwork.H_scaled.diagonal().array() *
                    qpwork.dw_aug.head(qpmodel.dim).array();
      qpwork.err.head(qpmodel.dim).noalias() -= Hdx;
      break;
  }
  qpwork.err.head(qpmodel.dim) -=
    qpresults.info.rho * qpwork.dw_aug.head(qpmodel.dim);

  // PERF: fuse {A, C}_scaled multiplication operations
  ATdy.noalias() = qpwork.A_scaled.transpose() *
                   qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
  qpwork.err.head(qpmodel.dim).noalias() -= ATdy;
  if (n_constraints > qpmodel.n_in) {
    // there are box constraints
    qpwork.active_part_z.tail(qpmodel.dim) = qpwork.dw_aug.head(qpmodel.dim);
    qpwork.active_part_z.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
    // ruiz.unscale_primal_in_place(VectorViewMut<T>{from_eigen,qpwork.active_part_z.tail(qpmodel.dim)});
  }
  for (isize i = 0; i < n_constraints; i++) {
    isize j = qpwork.current_bijection_map(i);
    if (j < qpwork.n_c) {
      if (i >= qpmodel.n_in) {
        // I_scaled * dz_box_scaled = unscale_primally(dz_box)
        qpwork.err(i - qpmodel.n_in) -=
          // qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
          // ruiz.delta(i-qpmodel.n_in);
          qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
          qpwork.i_scaled(i - qpmodel.n_in);
        // I_scaled * dx_scaled = dx_unscaled
        qpwork.err(qpmodel.dim + qpmodel.n_eq + j) -=
          (qpwork.active_part_z[i] -
           qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
             qpresults.info.mu_in);
      } else {
        qpwork.err.head(qpmodel.dim).noalias() -=
          qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
          qpwork.C_scaled.row(i);
        qpwork.err(qpmodel.dim + qpmodel.n_eq + j) -=
          (qpwork.C_scaled.row(i).dot(qpwork.dw_aug.head(qpmodel.dim)) -
           qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
             qpresults.info.mu_in);
      }
    }
  }
  Adx.noalias() = qpwork.A_scaled * qpwork.dw_aug.head(qpmodel.dim);
  qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).noalias() -= Adx;
  qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) +=
    qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) * qpresults.info.mu_eq;
}

template<typename T>
void
solve_linear_system(proxsuite::proxqp::dense::Vec<T>& dw,
                    const Model<T>& qpmodel,
                    Results<T>& qpresults,
                    Workspace<T>& qpwork,
                    const isize n_constraints,
                    const DenseBackend& dense_backend,
                    isize inner_pb_dim,
                    proxsuite::linalg::veg::dynstack::DynStackMut& stack)
{

  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT:
      qpwork.ldl.solve_in_place(dw.head(inner_pb_dim), stack);
      break;
    case DenseBackend::PrimalLDLT:
      // find dx
      dw.head(qpmodel.dim).noalias() += qpresults.info.mu_eq_inv *
                                        qpwork.A_scaled.transpose() *
                                        dw.segment(qpmodel.dim, qpmodel.n_eq);
      for (isize i = 0; i < n_constraints; i++) {
        isize j = qpwork.current_bijection_map(i);
        if (j < qpwork.n_c) {
          if (i >= qpmodel.n_in) {
            // box constraints
            dw(i - qpmodel.n_in) += dw(j + qpmodel.dim + qpmodel.n_eq) *
                                    qpwork.i_scaled(i - qpmodel.n_in);
          } else {
            // ineq constraints
            dw.head(qpmodel.dim) +=
              dw(j + qpmodel.dim + qpmodel.n_eq) * qpwork.C_scaled.row(i);
          }
        }
      }
      qpwork.ldl.solve_in_place(dw.head(qpmodel.dim), stack);
      // find dy
      dw.segment(qpmodel.dim, qpmodel.n_eq) -=
        qpresults.info.mu_eq_inv * dw.segment(qpmodel.dim, qpmodel.n_eq);
      dw.segment(qpmodel.dim, qpmodel.n_eq).noalias() +=
        qpresults.info.mu_eq_inv *
        (qpwork.A_scaled *
         dw.head(
           qpmodel.dim)); //- qpwork.rhs.segment(qpmodel.dim,qpmodel.n_eq));
      // find dz_J
      for (isize i = 0; i < n_constraints; i++) {
        isize j = qpwork.current_bijection_map(i);
        if (j < qpwork.n_c) {
          if (i >= qpmodel.n_in) {
            // box constraints
            dw(j + qpmodel.dim + qpmodel.n_eq) -=
              qpresults.info.mu_in_inv * (dw(j + qpmodel.dim + qpmodel.n_eq));
            dw(j + qpmodel.dim + qpmodel.n_eq) +=
              qpresults.info.mu_in_inv *
              (dw(
                i -
                qpmodel.n_in)); //- qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq));
          } else {
            // ineq constraints
            dw(j + qpmodel.dim + qpmodel.n_eq) -=
              qpresults.info.mu_in_inv * (dw(j + qpmodel.dim + qpmodel.n_eq));
            dw(j + qpmodel.dim + qpmodel.n_eq) +=
              qpresults.info.mu_in_inv *
              (qpwork.C_scaled.row(i).dot(dw.head(
                qpmodel.dim))); //- qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq));
          }
        }
      }
      break;
    case DenseBackend::Automatic:
      break;
  }
}

/*!
 * Performs iterative refinement for solving associated linear systems of PROXQP
 * algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param eps accuracy required for pursuing or not the iterative refinement.
 * @param inner_pb_dim dimension of the linear system.
 */
template<typename T>
void
iterative_solve_with_permut_fact( //
  const Settings<T>& qpsettings,
  const Model<T>& qpmodel,
  Results<T>& qpresults,
  Workspace<T>& qpwork,
  const isize n_constraints,
  const DenseBackend& dense_backend,
  const HessianType& hessian_type,
  T eps,
  isize inner_pb_dim)
{

  qpwork.err.setZero();
  i32 it = 0;
  i32 it_stability = 0;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };
  qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
  solve_linear_system(qpwork.dw_aug,
                      qpmodel,
                      qpresults,
                      qpwork,
                      n_constraints,
                      dense_backend,
                      inner_pb_dim,
                      stack);
  iterative_residual<T>(
    qpmodel, qpresults, qpwork, n_constraints, inner_pb_dim, hessian_type);

  ++it;
  T preverr = infty_norm(qpwork.err.head(inner_pb_dim));
  while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

    if (it >= qpsettings.nb_iterative_refinement) {
      break;
    }
    ++it;
    solve_linear_system(qpwork.err,
                        qpmodel,
                        qpresults,
                        qpwork,
                        n_constraints,
                        dense_backend,
                        inner_pb_dim,
                        stack);
    // qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), stack);
    qpwork.dw_aug.head(inner_pb_dim) += qpwork.err.head(inner_pb_dim);

    qpwork.err.head(inner_pb_dim).setZero();
    iterative_residual<T>(
      qpmodel, qpresults, qpwork, n_constraints, inner_pb_dim, hessian_type);

    if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
      it_stability += 1;

    } else {
      it_stability = 0;
    }
    if (it_stability == 2) {
      break;
    }
    preverr = infty_norm(qpwork.err.head(inner_pb_dim));
  }

  if (infty_norm(qpwork.err.head(inner_pb_dim)) >=
      std::max(eps, qpsettings.eps_refact)) {
    refactorize(qpmodel,
                qpresults,
                qpwork,
                n_constraints,
                dense_backend,
                qpresults.info.rho);
    it = 0;
    it_stability = 0;

    qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
    solve_linear_system(qpwork.dw_aug,
                        qpmodel,
                        qpresults,
                        qpwork,
                        n_constraints,
                        dense_backend,
                        inner_pb_dim,
                        stack);
    // qpwork.ldl.solve_in_place(qpwork.dw_aug.head(inner_pb_dim), stack);

    iterative_residual<T>(
      qpmodel, qpresults, qpwork, n_constraints, inner_pb_dim, hessian_type);

    preverr = infty_norm(qpwork.err.head(inner_pb_dim));
    ++it;
    while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

      if (it >= qpsettings.nb_iterative_refinement) {
        break;
      }
      ++it;
      solve_linear_system(qpwork.err,
                          qpmodel,
                          qpresults,
                          qpwork,
                          n_constraints,
                          dense_backend,
                          inner_pb_dim,
                          stack);
      // qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), stack);
      qpwork.dw_aug.head(inner_pb_dim) += qpwork.err.head(inner_pb_dim);

      qpwork.err.head(inner_pb_dim).setZero();
      iterative_residual<T>(
        qpmodel, qpresults, qpwork, n_constraints, inner_pb_dim, hessian_type);

      if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
        it_stability += 1;
      } else {
        it_stability = 0;
      }
      if (it_stability == 2) {
        break;
      }
      preverr = infty_norm(qpwork.err.head(inner_pb_dim));
    }
  }
  if (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps && qpsettings.verbose) {
    // std::cout << "after refact err " << err << std::endl;
    std::cout << "refact err " << infty_norm(qpwork.err.head(inner_pb_dim))
              << std::endl;
  }
  qpresults.info.iterative_residual = infty_norm(qpwork.err.head(inner_pb_dim));

  qpwork.rhs.head(inner_pb_dim).setZero();
}
/*!
 * BCL rule for updating penalization parameters and accuracy variables.
 *
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param primal_feasibility_lhs_new primal infeasibility.
 * @param bcl_eta_ext BCL variable measuring whether the precisely infeasibility
 * is too large or not.
 * @param bcl_eta_in BCL variable setting the accuracy required for solving an
 * associated subproblem.
 * @param bcl_eta_ext_init initial BCL bcl_eta_ext variable value.
 * @param eps_in_min minimal possible value for bcl_eta_in.
 * @param new_bcl_mu_in new value of the inequality constrained penalization
 * parameter.
 * @param new_bcl_mu_eq new value of the equality constrained penalization
 * parameter.
 * @param new_bcl_mu_in_inv new value of the inequality constrained penalization
 * parameter (inverse form).
 * @param new_bcl_mu_eq_inv new value of the equality constrained penalization
 * parameter (inverse form).
 */
template<typename T>
void
bcl_update(const Settings<T>& qpsettings,
           Results<T>& qpresults,
           Workspace<T>& qpwork,
           T& primal_feasibility_lhs_new,
           T& bcl_eta_ext,
           T& bcl_eta_in,

           T bcl_eta_ext_init,
           T eps_in_min,

           T& new_bcl_mu_in,
           T& new_bcl_mu_eq,
           T& new_bcl_mu_in_inv,
           T& new_bcl_mu_eq_inv

)
{
  if (primal_feasibility_lhs_new <= bcl_eta_ext ||
      qpresults.info.iter > qpsettings.safe_guard) {
    /* TO PUT IN DEBUG MODE
    if (qpsettings.verbose) {
            std::cout << "good step" << std::endl;
    }
    */
    bcl_eta_ext *= pow(qpresults.info.mu_in, qpsettings.beta_bcl);
    bcl_eta_in = std::max(bcl_eta_in * qpresults.info.mu_in, eps_in_min);
  } else {
    /* TO PUT IN DEBUG MODE
    if (qpsettings.verbose) {
            std::cout << "bad step" << std::endl;
    }
    */
    qpresults.y = qpwork.y_prev;
    qpresults.z = qpwork.z_prev;

    new_bcl_mu_in = std::max(qpresults.info.mu_in * qpsettings.mu_update_factor,
                             qpsettings.mu_min_in);
    new_bcl_mu_eq = std::max(qpresults.info.mu_eq * qpsettings.mu_update_factor,
                             qpsettings.mu_min_eq);
    new_bcl_mu_in_inv =
      std::min(qpresults.info.mu_in_inv * qpsettings.mu_update_inv_factor,
               qpsettings.mu_max_in_inv);
    new_bcl_mu_eq_inv =
      std::min(qpresults.info.mu_eq_inv * qpsettings.mu_update_inv_factor,
               qpsettings.mu_max_eq_inv);
    bcl_eta_ext = bcl_eta_ext_init * pow(new_bcl_mu_in, qpsettings.alpha_bcl);
    bcl_eta_in = std::max(new_bcl_mu_in, eps_in_min);
  }
}
/*!
 * Martinez rule for updating penalization parameters and accuracy variables.
 *
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param primal_feasibility_lhs_new primal infeasibility.
 * @param bcl_eta_ext BCL variable measuring whether the precisely infeasibility
 * is too large or not.
 * @param bcl_eta_in BCL variable setting the accuracy required for solving an
 * associated subproblem.
 * @param bcl_eta_ext_init initial BCL bcl_eta_ext variable value.
 * @param eps_in_min minimal possible value for bcl_eta_in.
 * @param new_bcl_mu_in new value of the inequality constrained penalization
 * parameter.
 * @param new_bcl_mu_eq new value of the equality constrained penalization
 * parameter.
 * @param new_bcl_mu_in_inv new value of the inequality constrained penalization
 * parameter (inverse form).
 * @param new_bcl_mu_eq_inv new value of the equality constrained penalization
 * parameter (inverse form).
 */
template<typename T>
void
Martinez_update(const Settings<T>& qpsettings,
                Results<T>& qpresults,
                T& primal_feasibility_lhs_new,
                T& primal_feasibility_lhs_old,
                T& bcl_eta_in,
                T eps_in_min,

                T& new_bcl_mu_in,
                T& new_bcl_mu_eq,
                T& new_bcl_mu_in_inv,
                T& new_bcl_mu_eq_inv

)
{
  bcl_eta_in = std::max(bcl_eta_in * 0.1, eps_in_min);
  if (primal_feasibility_lhs_new <= 0.95 * primal_feasibility_lhs_old) {
    /* TO PUT IN DEBUG MODE
    if (qpsettings.verbose) {
            std::cout << "good step" << std::endl;
    }
    */
  } else {
    /* TO PUT IN DEBUG MODE
    if (qpsettings.verbose) {
            std::cout << "bad step" << std::endl;
    }
    */
    new_bcl_mu_in = std::max(qpresults.info.mu_in * qpsettings.mu_update_factor,
                             qpsettings.mu_min_in);
    new_bcl_mu_eq = std::max(qpresults.info.mu_eq * qpsettings.mu_update_factor,
                             qpsettings.mu_min_eq);
    new_bcl_mu_in_inv =
      std::min(qpresults.info.mu_in_inv * qpsettings.mu_update_inv_factor,
               qpsettings.mu_max_in_inv);
    new_bcl_mu_eq_inv =
      std::min(qpresults.info.mu_eq_inv * qpsettings.mu_update_inv_factor,
               qpsettings.mu_max_eq_inv);
  }
}
/*!
 * Derives the stopping criterion value used by the Newton semismooth algorithm
 * to minimize the primal-dual augmented Lagrangian function.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
auto
compute_inner_loop_saddle_point(const Model<T>& qpmodel,
                                Results<T>& qpresults,
                                Workspace<T>& qpwork,
                                const Settings<T>& qpsettings) -> T
{

  qpwork.active_part_z =
    helpers::positive_part(qpwork.primal_residual_in_scaled_up) +
    helpers::negative_part(qpresults.si);
  switch (qpsettings.merit_function_type) {
    case MeritFunctionType::GPDAL:
      qpwork.active_part_z -=
        qpsettings.alpha_gpdal * qpresults.z *
        qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+

      // qpwork.active_part_z.head(qpmodel.n_in) -=
      //   qpsettings.alpha_gpdal * qpresults.z.head(qpmodel.n_in) *
      //   qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+
      // if (box_constraints){
      //   qpwork.active_part_z.tail(qpmodel.dim) -=
      //     qpsettings.alpha_gpdal * qpresults.z.tail(qpmodel.dim) *
      //     qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+
      // }
      break;
    case MeritFunctionType::PDAL:
      qpwork.active_part_z -=
        qpresults.z *
        qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+
                              // + [Cx-l+z_prev*mu_in]- - z*mu_in
      // qpwork.active_part_z.head(qpmodel.n_in) -=
      //   qpresults.z.head(qpmodel.n_in) *
      //   qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+
      //                         // + [Cx-l+z_prev*mu_in]- - z*mu_in
      // if (box_constraints){
      //   qpwork.active_part_z.tail(qpmodel.dim) -=
      //     qpresults.z.tail(qpmodel.dim) *
      //     qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+
      // }
      break;
  }

  T err = infty_norm(qpwork.active_part_z);
  qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) = qpresults.se;
  // qpwork.primal_residual_eq_scaled; // contains now Ax-b-(y-y_prev)/mu

  T prim_eq_e = infty_norm(
    qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)); // ||Ax-b-(y-y_prev)/mu||
  err = std::max(err, prim_eq_e);
  T dual_e =
    infty_norm(qpwork.dual_residual_scaled); // contains ||Hx + rho(x-xprev) +
                                             // g + Aty + Ctz||
  err = std::max(err, dual_e);

  return err;
}
/*!
 * Derives the Newton semismooth step.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param eps accuracy required for solving the subproblem.
 */
template<typename T>
void
primal_dual_semi_smooth_newton_step(const Settings<T>& qpsettings,
                                    const Model<T>& qpmodel,
                                    Results<T>& qpresults,
                                    Workspace<T>& qpwork,
                                    const bool box_constraints,
                                    const isize n_constraints,
                                    const DenseBackend& dense_backend,
                                    const HessianType& hessian_type,
                                    T eps)
{

  /* MUST BE
   *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
   *  primal_residual_eq_scaled = Ax-b+mu_eq (y_prev-y)
   *  primal_residual_in_scaled_up = Cx-u+mu_in(z_prev)
   *  primal_residual_in_scaled_low = Cx-l+mu_in(z_prev)
   */
  qpwork.active_set_up.array() =
    (qpwork.primal_residual_in_scaled_up.array() >= 0);
  // primal_residual_in_scaled_low = Cx-l+mu_in(z_prev) + mu_in(alpha_gdpal - 1)
  // * z
  qpwork.active_set_low.array() = (qpresults.si.array() <= 0);
  qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;

  isize numactive_inequalities = qpwork.active_inequalities.count();
  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
  qpwork.rhs.setZero();
  qpwork.dw_aug.setZero();
  linesearch::active_set_change(
    qpmodel, qpresults, dense_backend, n_constraints, qpwork);

  qpwork.rhs.head(qpmodel.dim) = -qpwork.dual_residual_scaled;

  if (box_constraints) {
    // use active_part_z as tmp variable in order to unscale primarilly z
    // as I_scaled.T * z_scaled = unscale_primarilly_(z_scaled)
    qpwork.active_part_z.tail(qpmodel.dim) = qpresults.z.tail(qpmodel.dim);
    qpwork.active_part_z.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
    // ruiz.unscale_primal_in_place(VectorViewMut<T>{from_eigen,qpwork.active_part_z.tail(qpmodel.dim)});
  }

  qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = -qpresults.se;
  // -qpwork.primal_residual_eq_scaled;
  switch (qpsettings.merit_function_type) {
    case MeritFunctionType::GPDAL:
      for (isize i = 0; i < n_constraints; i++) {
        isize j = qpwork.current_bijection_map(i);
        if (j < qpwork.n_c) {
          if (qpwork.active_set_up(i)) {
            qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
              -qpwork.primal_residual_in_scaled_up(i) +
              qpresults.z(i) * qpresults.info.mu_in * qpsettings.alpha_gpdal;
          } else if (qpwork.active_set_low(i)) {
            qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
              -qpresults.si(i) +
              qpresults.z(i) * qpresults.info.mu_in * qpsettings.alpha_gpdal;
          }
        } else {
          // unactive unrelevant columns
          if (i >= qpmodel.n_in) {
            qpwork.rhs(i - qpmodel.n_in) += qpwork.active_part_z(i);
          } else {
            qpwork.rhs.head(qpmodel.dim) +=
              qpresults.z(i) * qpwork.C_scaled.row(i);
          }
        }
      }
      break;
    case MeritFunctionType::PDAL:
      for (isize i = 0; i < n_constraints; i++) {
        isize j = qpwork.current_bijection_map(i);
        if (j < qpwork.n_c) {
          if (qpwork.active_set_up(i)) {
            qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
              -qpwork.primal_residual_in_scaled_up(i) +
              qpresults.z(i) * qpresults.info.mu_in;
          } else if (qpwork.active_set_low(i)) {
            qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
              -qpresults.si(i) + qpresults.z(i) * qpresults.info.mu_in;
          }
        } else {
          if (i >= qpmodel.n_in) {
            // unactive unrelevant columns
            qpwork.rhs(i - qpmodel.n_in) += qpwork.active_part_z(i);
          } else {
            qpwork.rhs.head(qpmodel.dim) +=
              qpresults.z(i) * qpwork.C_scaled.row(i);
          }
        }
      }
      break;
  }
  iterative_solve_with_permut_fact( //
    qpsettings,
    qpmodel,
    qpresults,
    qpwork,
    n_constraints,
    dense_backend,
    hessian_type,
    eps,
    inner_pb_dim);

  // use active_part_z as a temporary variable to derive unpermutted dz step
  for (isize j = 0; j < n_constraints; ++j) {
    isize i = qpwork.current_bijection_map(j);
    if (i < qpwork.n_c) {
      qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i);
    } else {
      qpwork.active_part_z(j) = -qpresults.z(j);
    }
  }
  qpwork.dw_aug.tail(n_constraints) = qpwork.active_part_z;
}
/*!
 * Performs the Newton semismooth algorithm to minimize the primal-dual
 * augmented Lagrangian function used by PROXQP algorithm.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param eps_int accuracy required for solving the subproblem.
 */
template<typename T>
void
primal_dual_newton_semi_smooth(const Settings<T>& qpsettings,
                               const Model<T>& qpmodel,
                               Results<T>& qpresults,
                               Workspace<T>& qpwork,
                               const bool box_constraints,
                               const isize n_constraints,
                               preconditioner::RuizEquilibration<T>& ruiz,
                               const DenseBackend& dense_backend,
                               const HessianType& hessian_type,
                               T eps_int)
{

  /* MUST CONTAIN IN ENTRY WITH x = x_prev ; y = y_prev ; z = z_prev
   *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
   *  primal_residual_eq_scaled = Ax-b+mu_eq (y_prev-y)
   *  primal_residual_in_scaled_up = Cx-u+mu_in(z_prev)
   *  primal_residual_in_scaled_low = Cx-l+mu_in(z_prev)
   */
  /* for debug
  if (qpsettings.verbose) {
          std::cout << "---- inner iteration    inner error    alpha ----" <<
  std::endl;
  }
  */
  T err_in = 1.e6;

  for (i64 iter = 0; iter <= qpsettings.max_iter_in; ++iter) {

    if (iter == qpsettings.max_iter_in) {
      qpresults.info.iter += qpsettings.max_iter_in + 1;
      break;
    }
    proxsuite::linalg::veg::dynstack::DynStackMut stack{
      proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
    };
    primal_dual_semi_smooth_newton_step<T>(qpsettings,
                                           qpmodel,
                                           qpresults,
                                           qpwork,
                                           box_constraints,
                                           n_constraints,
                                           dense_backend,
                                           hessian_type,
                                           eps_int);

    auto& Hdx = qpwork.Hdx;
    auto& Adx = qpwork.Adx;
    auto& Cdx = qpwork.Cdx;
    auto& ATdy = qpwork.CTz;
    auto dx = qpwork.dw_aug.head(qpmodel.dim);
    auto dy = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
    auto dz = qpwork.dw_aug.tail(n_constraints);
    LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);
    if (qpmodel.n_in > 0) {
      Cdx.head(qpmodel.n_in).noalias() = qpwork.C_scaled * dx;
      CTdz.noalias() = qpwork.C_scaled.transpose() * dz.head(qpmodel.n_in);
    }
    if (box_constraints) {
      // use active_part_z as tmp variable in order to unscale primarilly dz
      // as I_scaled.T * dz_scaled = unscale_primarilly_(dz_scaled)

      // qpwork.active_part_z.tail(qpmodel.dim) = dz.tail(qpmodel.dim);
      // ruiz.unscale_primal_in_place(VectorViewMut<T>{from_eigen,qpwork.active_part_z.tail(qpmodel.dim)});
      // CTdz.noalias() += qpwork.active_part_z.tail(qpmodel.dim);
      // Cdx.tail(qpmodel.dim) = dx;
      // // I_scaled * dx_scaled = dx_unscaled
      // ruiz.unscale_primal_in_place(VectorViewMut<T>{from_eigen,
      // Cdx.tail(qpmodel.dim)});

      qpwork.active_part_z.tail(qpmodel.dim) = dz.tail(qpmodel.dim);
      qpwork.active_part_z.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
      CTdz.noalias() += qpwork.active_part_z.tail(qpmodel.dim);

      Cdx.tail(qpmodel.dim) = dx;
      Cdx.tail(qpmodel.dim).array() *= qpwork.i_scaled.array();
    }
    switch (qpsettings.merit_function_type) {
      case MeritFunctionType::GPDAL:
        Cdx.noalias() +=
          (qpsettings.alpha_gpdal - 1.) * qpresults.info.mu_in * dz;
        break;
      case MeritFunctionType::PDAL:
        break;
    }
    if (qpmodel.n_in > 0 || box_constraints) {
      linesearch::primal_dual_ls(
        qpmodel, qpresults, qpwork, qpsettings, n_constraints);
    }
    auto alpha = qpwork.alpha;
    if (infty_norm(alpha * qpwork.dw_aug) < 1.E-11 && iter > 0) {
      qpresults.info.iter += iter + 1;
      break;
      /* to put in debuger mode
      if (qpsettings.verbose) {
              std::cout << "infty_norm(alpha_step * dx) "
                                                      << infty_norm(alpha *
      qpwork.dw_aug) << std::endl;
      }
      */
    }
    qpresults.x += alpha * dx;

    // contains now :  C(x+alpha dx)-u + z_prev * mu_in
    qpwork.primal_residual_in_scaled_up += alpha * Cdx;

    // contains now :  C(x+alpha dx)-l + z_prev * mu_in
    qpresults.si += alpha * Cdx;

    // qpwork.primal_residual_eq_scaled +=
    qpresults.se += alpha * (Adx - qpresults.info.mu_eq * dy);
    qpresults.y += alpha * dy;
    qpresults.z += alpha * dz;
    switch (hessian_type) {
      case HessianType::Zero:
        qpwork.dual_residual_scaled +=
          alpha * (qpresults.info.rho * dx + ATdy + CTdz);
        break;
      case HessianType::Dense:
        qpwork.dual_residual_scaled +=
          alpha * (qpresults.info.rho * dx + Hdx + ATdy + CTdz);
        break;
      case HessianType::Diagonal:
        qpwork.dual_residual_scaled +=
          alpha * (qpresults.info.rho * dx + Hdx + ATdy + CTdz);
        break;
    }

    err_in = dense::compute_inner_loop_saddle_point(
      qpmodel, qpresults, qpwork, qpsettings);
    /* for debug
    if (qpsettings.verbose) {
            std::cout << "           " << iter << "              " <<
    std::setprecision(2) << err_in
                                                    << "         "  << alpha <<
    std::endl;
    }
    */
    if (qpsettings.verbose) {
      std::cout << "\033[1;34m[inner iteration " << iter + 1 << "]\033[0m"
                << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| inner residual=" << err_in << " | alpha=" << alpha
                << std::endl;
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
        if (!qpsettings.primal_infeasibility_solving) {
          qpresults.info.iter += iter + 1;
          break;
        }
      } else if (is_dual_infeasible) {
        qpresults.info.status = QPSolverOutput::PROXQP_DUAL_INFEASIBLE;
        qpresults.info.iter += iter + 1;
        break;
      }
    }
    if (err_in <= eps_int) {
      qpresults.info.iter += iter + 1;
      break;
    }
  }
  /* to put in debuger mode
  if (qpsettings.verbose) {
    if (err_in > eps_int){
          std::cout << " inner loop residual is to high! Its value is equal to "
  << err_in << ", while it should be inferior to: "  << eps_int << std::endl;
    }
  }
  */
}
/*!
 * Executes the PROXQP algorithm.
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
  /*** TEST WITH MATRIX FULL OF NAN FOR DEBUG
    static constexpr Layout layout = rowmajor;
    static constexpr auto DYN = Eigen::Dynamic;
  using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
  RowMat test(2,2); // test it is full of nan for debug
  std::cout << "test " << test << std::endl;
  */
  PROXSUITE_EIGEN_MALLOC_NOT_ALLOWED();
  isize n_constraints(qpmodel.n_in);
  if (box_constraints) {
    n_constraints += qpmodel.dim;
  }
  if (qpsettings.compute_timings) {
    qpwork.timer.stop();
    qpwork.timer.start();
  }
  if (qpsettings.verbose) {
    dense::print_setup_header(qpsettings,
                              qpresults,
                              qpmodel,
                              box_constraints,
                              dense_backend,
                              hessian_type);
  }
  // std::cout << "qpwork.dirty " << qpwork.dirty << std::endl;
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
    switch (qpsettings.initial_guess) {
      case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
        compute_equality_constrained_initial_guess(qpwork,
                                                   qpsettings,
                                                   qpmodel,
                                                   n_constraints,
                                                   dense_backend,
                                                   hessian_type,
                                                   qpresults);
        break;
      }
      case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
        //!\ TODO in a quicker way
        qpwork.n_c = 0;
        for (isize i = 0; i < n_constraints; i++) {
          if (qpresults.z[i] != 0) {
            qpwork.active_inequalities[i] = true;
          } else {
            qpwork.active_inequalities[i] = false;
          }
        }
        linesearch::active_set_change(
          qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        break;
      }
      case InitialGuessStatus::NO_INITIAL_GUESS: {
        break;
      }
      case InitialGuessStatus::WARM_START: {
        //!\ TODO in a quicker way
        qpwork.n_c = 0;
        for (isize i = 0; i < n_constraints; i++) {
          if (qpresults.z[i] != 0) {
            qpwork.active_inequalities[i] = true;
          } else {
            qpwork.active_inequalities[i] = false;
          }
        }
        linesearch::active_set_change(
          qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        break;
      }
      case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
        // keep workspace and results solutions except statistics
        // std::cout << "i use previous solution" << std::endl;
        // meaningful for when one wants to warm start with previous result with
        // the same QP model
        break;
      }
    }
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
        qpwork.n_c = 0;
        for (isize i = 0; i < n_constraints; i++) {
          if (qpresults.z[i] != 0) {
            qpwork.active_inequalities[i] = true;
          } else {
            qpwork.active_inequalities[i] = false;
          }
        }
        linesearch::active_set_change(
          qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        break;
      }
      case InitialGuessStatus::NO_INITIAL_GUESS: {
        setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
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
        qpwork.n_c = 0;
        for (isize i = 0; i < n_constraints; i++) {
          if (qpresults.z[i] != 0) {
            qpwork.active_inequalities[i] = true;
          } else {
            qpwork.active_inequalities[i] = false;
          }
        }
        linesearch::active_set_change(
          qpmodel, qpresults, dense_backend, n_constraints, qpwork);
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
          qpwork.n_c = 0;
          for (isize i = 0; i < n_constraints; i++) {
            if (qpresults.z[i] != 0) {
              qpwork.active_inequalities[i] = true;
            } else {
              qpwork.active_inequalities[i] = false;
            }
          }
          linesearch::active_set_change(
            qpmodel, qpresults, dense_backend, n_constraints, qpwork);
          break;
        }
      }
    }
  }
  T bcl_eta_ext_init = pow(T(0.1), qpsettings.alpha_bcl);
  T bcl_eta_ext = bcl_eta_ext_init;
  T bcl_eta_in(1);
  T eps_in_min = std::min(qpsettings.eps_abs, T(1.E-9));

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

    // compute primal residual

    // PERF: fuse matrix product computations in global_{primal, dual}_residual
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

    T new_bcl_mu_in(qpresults.info.mu_in);
    T new_bcl_mu_eq(qpresults.info.mu_eq);
    T new_bcl_mu_in_inv(qpresults.info.mu_in_inv);
    T new_bcl_mu_eq_inv(qpresults.info.mu_eq_inv);

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
        // EigenAllowAlloc _{};
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
      std::cout << "\033[1;32m[outer iteration " << iter + 1 << "]\033[0m"
                << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | mu_in=" << qpresults.info.mu_in
                << " | rho=" << qpresults.info.rho << std::endl;
      ruiz.scale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
      ruiz.scale_dual_in_place_eq(VectorViewMut<T>{ from_eigen, qpresults.y });
      ruiz.scale_dual_in_place_in(
        VectorViewMut<T>{ from_eigen, qpresults.z.head(qpmodel.n_in) });
      if (box_constraints) {
        ruiz.scale_box_dual_in_place_in(
          VectorViewMut<T>{ from_eigen, qpresults.z.tail(qpmodel.dim) });
      }
    }
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
    qpresults.info.iter_ext += 1; // We start a new external loop update

    qpwork.x_prev = qpresults.x;
    qpwork.y_prev = qpresults.y;
    qpwork.z_prev = qpresults.z;

    // primal dual version from gill and robinson

    ruiz.scale_primal_residual_in_place_in(
      VectorViewMut<T>{ from_eigen,
                        qpwork.primal_residual_in_scaled_up.head(
                          qpmodel.n_in) }); // contains now scaled(Cx)
    if (box_constraints) {
      ruiz.scale_box_primal_residual_in_place_in(
        VectorViewMut<T>{ from_eigen,
                          qpwork.primal_residual_in_scaled_up.tail(
                            qpmodel.dim) }); // contains now scaled(x)
    }
    qpwork.primal_residual_in_scaled_up +=
      qpwork.z_prev *
      qpresults.info.mu_in; // contains now scaled(Cx+z_prev*mu_in)
    switch (qpsettings.merit_function_type) {
      case MeritFunctionType::GPDAL:
        qpwork.primal_residual_in_scaled_up +=
          (qpsettings.alpha_gpdal - 1.) * qpresults.info.mu_in * qpresults.z;
        break;
      case MeritFunctionType::PDAL:
        break;
    }
    qpresults.si = qpwork.primal_residual_in_scaled_up;
    qpwork.primal_residual_in_scaled_up.head(qpmodel.n_in) -=
      qpwork.u_scaled; // contains now scaled(Cx-u+z_prev*mu_in)
    qpresults.si.head(qpmodel.n_in) -=
      qpwork.l_scaled; // contains now scaled(Cx-l+z_prev*mu_in)
    if (box_constraints) {
      // qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) -=
      //   qpmodel.u_box; // contains now scaled(Cx-u+z_prev*mu_in)
      // qpwork.primal_residual_in_scaled_low.tail(qpmodel.dim) -=
      //   qpmodel.l_box; // contains now scaled(Cx-l+z_prev*mu_in)

      qpwork.primal_residual_in_scaled_up.tail(qpmodel.dim) -=
        qpwork.u_box_scaled; // contains now scaled(Cx-u+z_prev*mu_in)
      qpresults.si.tail(qpmodel.dim) -=
        qpwork.l_box_scaled; // contains now scaled(Cx-l+z_prev*mu_in)
    }

    primal_dual_newton_semi_smooth(qpsettings,
                                   qpmodel,
                                   qpresults,
                                   qpwork,
                                   box_constraints,
                                   n_constraints,
                                   ruiz,
                                   dense_backend,
                                   hessian_type,
                                   bcl_eta_in);

    if ((qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE &&
         !qpsettings.primal_infeasibility_solving) ||
        qpresults.info.status == QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      // certificate of infeasibility
      qpresults.x = qpwork.dw_aug.head(qpmodel.dim);
      qpresults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
      qpresults.z = qpwork.dw_aug.tail(n_constraints);
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
    }
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
    if (qpsettings.bcl_update) {
      bcl_update(qpsettings,
                 qpresults,
                 qpwork,
                 primal_feasibility_lhs_new,
                 bcl_eta_ext,
                 bcl_eta_in,
                 bcl_eta_ext_init,
                 eps_in_min,

                 new_bcl_mu_in,
                 new_bcl_mu_eq,
                 new_bcl_mu_in_inv,
                 new_bcl_mu_eq_inv);
    } else {
      Martinez_update(qpsettings,
                      qpresults,
                      primal_feasibility_lhs_new,
                      primal_feasibility_lhs,
                      bcl_eta_in,
                      eps_in_min,
                      new_bcl_mu_in,
                      new_bcl_mu_eq,
                      new_bcl_mu_in_inv,
                      new_bcl_mu_eq_inv);
    }
    // COLD RESTART

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

    if (primal_feasibility_lhs_new >= primal_feasibility_lhs &&
        dual_feasibility_lhs_new >= dual_feasibility_lhs &&
        qpresults.info.mu_in <= T(1e-5)) {
      /* to put in debuger mode
      if (qpsettings.verbose) {
              std::cout << "cold restart" << std::endl;
      }
      */

      new_bcl_mu_in = qpsettings.cold_reset_mu_in;
      new_bcl_mu_eq = qpsettings.cold_reset_mu_eq;
      new_bcl_mu_in_inv = qpsettings.cold_reset_mu_in_inv;
      new_bcl_mu_eq_inv = qpsettings.cold_reset_mu_eq_inv;
    }

    /// effective mu upddate

    if (qpresults.info.mu_in != new_bcl_mu_in ||
        qpresults.info.mu_eq != new_bcl_mu_eq) {
      {
        ++qpresults.info.mu_updates;
      }
      mu_update(qpmodel,
                qpresults,
                qpwork,
                n_constraints,
                dense_backend,
                new_bcl_mu_eq,
                new_bcl_mu_in);
    }

    qpresults.info.mu_eq = new_bcl_mu_eq;
    qpresults.info.mu_in = new_bcl_mu_in;
    qpresults.info.mu_eq_inv = new_bcl_mu_eq_inv;
    qpresults.info.mu_in_inv = new_bcl_mu_in_inv;
  }

  ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
  ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{ from_eigen, qpresults.y });
  ruiz.unscale_dual_in_place_in(
    VectorViewMut<T>{ from_eigen, qpresults.z.head(qpmodel.n_in) });
  if (box_constraints) {
    ruiz.unscale_box_dual_in_place_in(
      VectorViewMut<T>{ from_eigen, qpresults.z.tail(qpmodel.dim) });
  }
  if (qpsettings.primal_infeasibility_solving &&
      qpresults.info.status == QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
    ruiz.unscale_primal_residual_in_place_eq(
      VectorViewMut<T>{ from_eigen, qpresults.se });
    ruiz.unscale_primal_residual_in_place_in(
      VectorViewMut<T>{ from_eigen, qpresults.si.head(qpmodel.n_in) });
    if (box_constraints) {
      ruiz.unscale_box_primal_residual_in_place_in(
        VectorViewMut<T>{ from_eigen, qpresults.si.tail(qpmodel.dim) });
    }
  }

  {
    // EigenAllowAlloc _{};
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

  if (qpsettings.compute_timings) {
    qpresults.info.solve_time = qpwork.timer.elapsed().user; // in nanoseconds
    qpresults.info.run_time =
      qpresults.info.solve_time + qpresults.info.setup_time;
  }

  if (qpsettings.verbose) {
    std::cout << "-------------------SOLVER STATISTICS-------------------"
              << std::endl;
    std::cout << "outer iter:   " << qpresults.info.iter_ext << std::endl;
    std::cout << "total iter:   " << qpresults.info.iter << std::endl;
    std::cout << "mu updates:   " << qpresults.info.mu_updates << std::endl;
    std::cout << "rho updates:  " << qpresults.info.rho_updates << std::endl;
    std::cout << "objective:    " << qpresults.info.objValue << std::endl;
    switch (qpresults.info.status) {
      case QPSolverOutput::PROXQP_SOLVED: {
        std::cout << "status:       "
                  << "Solved" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_MAX_ITER_REACHED: {
        std::cout << "status:       "
                  << "Maximum number of iterations reached" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE: {
        std::cout << "status:       "
                  << "Primal infeasible" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_DUAL_INFEASIBLE: {
        std::cout << "status:       "
                  << "Dual infeasible" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: {
        std::cout << "status:       "
                  << "Solved closest primal feasible" << std::endl;
        break;
      }
      case QPSolverOutput::PROXQP_NOT_RUN: {
        std::cout << "status:       "
                  << "Solver not run" << std::endl;
        break;
      }
    }

    if (qpsettings.compute_timings)
      std::cout << "run time:     " << qpresults.info.solve_time << std::endl;
    std::cout << "--------------------------------------------------------"
              << std::endl;
  }
  qpwork.dirty = true;
  qpwork.is_initialized = true; // necessary because we call workspace cleanup

  assert(!std::isnan(qpresults.info.pri_res));
  assert(!std::isnan(qpresults.info.dua_res));
  assert(!std::isnan(qpresults.info.duality_gap));

  PROXSUITE_EIGEN_MALLOC_ALLOWED();
}

} // namespace dense

} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_SOLVER_HPP */
