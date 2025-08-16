//
// Copyright (c) 2022-2023 INRIA
//
/** \file */
#ifndef PROXSUITE_COMMON_DENSE_PRINTS_HPP
#define PROXSUITE_COMMON_DENSE_PRINTS_HPP

#include "proxsuite/common/results.hpp"
#include "proxsuite/common/utils/prints.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/common/dense/workspace.hpp"

namespace proxsuite {
namespace common {
namespace dense {

using proxsuite::common::Results;
using proxsuite::common::Settings;
using proxsuite::common::Solver;
using proxsuite::common::dense::Workspace;

template<typename T>
void
print_setup_header(const Settings<T>& settings,
                   const Results<T>& results,
                   const proxsuite::proxqp::dense::Model<T>& model,
                   const bool box_constraints,
                   const DenseBackend& dense_backend,
                   const HessianType& hessian_type,
                   const Solver solver)
{

  proxsuite::common::print_preambule(solver);

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
  switch (solver) {
    case Solver::PROXQP: {
      break;
    }
    case Solver::OSQP: {
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
        std::cout << "          polish_refine_iter: "
                  << settings.polish_refine_iter << ". \n"
                  << std::endl;
      } else {
        std::cout << "          polishing: off. \n" << std::endl;
      }
      break;
    }
  }
}

} // namespace dense
} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_COMMON_DENSE_PRINTS_HPP */
