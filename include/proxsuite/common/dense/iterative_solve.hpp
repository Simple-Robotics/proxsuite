//
// Copyright (c) 2025 INRIA
//
/**
 * @file solver.hpp
 */

#ifndef PROXSUITE_COMMON_DENSE_ITERATIVE_SOLVE_HPP
#define PROXSUITE_COMMON_DENSE_ITERATIVE_SOLVE_HPP

#include <Eigen/Sparse>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>
#include <proxsuite/linalg/dense/ldlt.hpp>
#include "proxsuite/common/settings.hpp"
#include "proxsuite/common/results.hpp"
#include "proxsuite/common/dense/workspace.hpp"
#include "proxsuite/common/dense/model.hpp"
#include <iostream>

namespace proxsuite {
namespace common {
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
solve_linear_system(Vec<T>& dw,
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

} // namespace dense
} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_COMMON_DENSE_ITERATIVE_SOLVE_HPP */
