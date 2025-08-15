//
// Copyright (c) 2022-2024 INRIA
//
/**
 * @file utils.hpp
 */
#ifndef PROXSUITE_OSQP_DENSE_UTILS_HPP
#define PROXSUITE_OSQP_DENSE_UTILS_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <type_traits>

#include "proxsuite/common/status.hpp"
#include "proxsuite/helpers/common.hpp"
#include "proxsuite/common/dense/views.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include <proxsuite/proxqp/dense/model.hpp>
#include <proxsuite/proxqp/results.hpp>
#include <proxsuite/osqp/utils/prints.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>

namespace proxsuite {
namespace osqp {
namespace dense {

using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::dense;

template<typename T>
void
print_setup_header(const Settings<T>& settings,
                   const Results<T>& results,
                   const Model<T>& model,
                   const bool box_constraints,
                   const DenseBackend& dense_backend,
                   const HessianType& hessian_type)
{

  proxsuite::osqp::print_preambule();

  // Print variables and constraints
  std::cout << "problem:  " << std::noshowpos << std::endl;
  std::cout << "          variables n = " << model.dim
            << ", equality constraints n_eq = " << model.n_eq << ",\n"
            << "          inequality constraints n_in = " << model.n_in
            << std::endl;

  // Print Settings
  std::cout << "settings: " << std::endl;
  std::cout << "          backend = dense," << std::endl;
  std::cout << "          eps_abs = " << settings.eps_abs
            << " eps_rel = " << settings.eps_rel << std::endl;
  std::cout << "          eps_prim_inf = " << settings.eps_primal_inf
            << ", eps_dual_inf = " << settings.eps_dual_inf << "," << std::endl;

  std::cout << "          rho = " << results.info.rho
            << ", mu_eq = " << results.info.mu_eq
            << ", mu_in = " << results.info.mu_in << "," << std::endl;
  std::cout << "          max_iter = " << settings.max_iter
            << ", max_iter_in = " << settings.max_iter_in << "," << std::endl;
  if (box_constraints) {
    std::cout << "          box constraints: on, " << std::endl;
  } else {
    std::cout << "          box constraints: off, " << std::endl;
  }
  switch (dense_backend) {
    case DenseBackend::PrimalDualLDLT:
      std::cout << "          dense backend: PrimalDualLDLT, " << std::endl;
      break;
    case DenseBackend::PrimalLDLT:
      std::cout << "          dense backend: PrimalLDLT, " << std::endl;
      break;
    case DenseBackend::Automatic:
      break;
  }
  switch (hessian_type) {
    case HessianType::Dense:
      std::cout << "          problem type: Quadratic Program, " << std::endl;
      break;
    case HessianType::Zero:
      std::cout << "          problem type: Linear Program, " << std::endl;
      break;
    case HessianType::Diagonal:
      std::cout
        << "          problem type: Quadratic Program with diagonal Hessian, "
        << std::endl;
      break;
  }
  if (settings.compute_preconditioner) {
    std::cout << "          scaling: on, " << std::endl;
  } else {
    std::cout << "          scaling: off, " << std::endl;
  }
  if (settings.compute_timings) {
    std::cout << "          timings: on, " << std::endl;
  } else {
    std::cout << "          timings: off, " << std::endl;
  }
  switch (settings.initial_guess) {
    case InitialGuessStatus::WARM_START:
      std::cout << "          initial guess: warm start. \n" << std::endl;
      break;
    case InitialGuessStatus::NO_INITIAL_GUESS:
      std::cout << "          initial guess: no initial guess. \n" << std::endl;
      break;
    case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:
      std::cout
        << "          initial guess: warm start with previous result. \n"
        << std::endl;
      break;
    case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:
      std::cout
        << "          initial guess: cold start with previous result. \n"
        << std::endl;
      break;
    case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:
      std::cout
        << "          initial guess: equality constrained initial guess. \n"
        << std::endl;
  }
  if (settings.adaptive_mu) {
    std::cout << "          adaptive_mu: on, " << std::endl;
    std::cout << "          adaptive_mu_interval: "
              << settings.adaptive_mu_interval << ", " << std::endl;
    std::cout << "          adaptive_mu_tolerance: "
              << settings.adaptive_mu_tolerance << ". \n"
              << std::endl;
  } else {
    std::cout << "          adaptive_mu: off. \n" << std::endl;
  }
  if (settings.polishing) {
    std::cout << "          polishing: on, " << std::endl;
    std::cout << "          delta: " << settings.delta_osqp << ", "
              << std::endl;
    std::cout << "          polish_refine_iter: " << settings.polish_refine_iter
              << ". \n"
              << std::endl;
  } else {
    std::cout << "          polishing: off. \n" << std::endl;
  }
}

template<typename T>
void
setup_factorization_complete_kkt(Results<T>& qpresults,
                                 const Model<T>& qpmodel,
                                 Workspace<T>& qpwork,
                                 const isize n_constraints,
                                 const DenseBackend& dense_backend)
{
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, qpwork.ldl_stack.as_mut()
  };

  // Delete columns (from potential previous solve)
  if (qpwork.dirty == true) {
    auto _planned_to_delete = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, isize(n_constraints));
    isize* planned_to_delete = _planned_to_delete.ptr_mut();

    for (isize i = 0; i < n_constraints; i++) {
      planned_to_delete[i] = qpmodel.dim + qpmodel.n_eq + i;
    }

    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT: {
        qpwork.ldl.delete_at(planned_to_delete, n_constraints, stack);
      } break;
      case DenseBackend::PrimalLDLT:
        break;
      case DenseBackend::Automatic:
        break;
    }
  }

  // Add columns
  {
    T mu_in_neg(-qpresults.info.mu_in);
    switch (dense_backend) {
      case DenseBackend::PrimalDualLDLT: {
        isize n = qpmodel.dim;
        isize n_eq = qpmodel.n_eq;
        LDLT_TEMP_MAT_UNINIT(
          T, new_cols, n + n_eq + n_constraints, n_constraints, stack);

        for (isize k = 0; k < n_constraints; ++k) {
          auto col = new_cols.col(k);
          if (k >= qpmodel.n_in) {
            col.head(n).setZero();
            col[k - qpmodel.n_in] = qpwork.i_scaled[k - qpmodel.n_in];
          } else {
            col.head(n) = (qpwork.C_scaled.row(k));
          }
          col.tail(n_eq + n_constraints).setZero();
          col[n + n_eq + k] = mu_in_neg;
        }
        qpwork.ldl.insert_block_at(n + n_eq, new_cols, stack);
      } break;
      case DenseBackend::PrimalLDLT:
        break;
      case DenseBackend::Automatic:
        break;
    }
  }

  qpwork.n_c = n_constraints;
}

} // namespace dense
} // namespace osqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_DENSE_UTILS_HPP */
