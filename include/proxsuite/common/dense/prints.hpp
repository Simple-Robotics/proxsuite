//
// Copyright (c) 2022-2025 INRIA
//
/** \file */
#ifndef PROXSUITE_COMMON_DENSE_PRINTS_HPP
#define PROXSUITE_COMMON_DENSE_PRINTS_HPP

#include "proxsuite/common/dense/views.hpp"
#include "proxsuite/common/results.hpp"
#include "proxsuite/common/status.hpp"
#include "proxsuite/common/utils/prints.hpp"
#include "proxsuite/common/dense/model.hpp"
#include "proxsuite/common/dense/workspace.hpp"
#include "proxsuite/common/dense/preconditioner/ruiz.hpp"
#include <iomanip>

namespace proxsuite {
namespace common {
namespace dense {

template<typename T>
void
print_setup_header(const Settings<T>& qpsettings,
                   const Results<T>& qpresults,
                   const Model<T>& qpmodel,
                   const bool box_constraints,
                   const DenseBackend& dense_backend,
                   const HessianType& hessian_type,
                   const QPSolver solver)
{

  print_preambule(solver);

  // Print variables and constraints
  std::cout << "problem:  " << std::noshowpos << std::endl;
  std::cout << "          variables n = " << qpmodel.dim
            << ", equality constraints n_eq = " << qpmodel.n_eq << ",\n"
            << "          inequality constraints n_in = " << qpmodel.n_in
            << std::endl;

  // Print Settings
  std::cout << "settings: " << std::endl;
  std::cout << "          backend = dense," << std::endl;
  std::cout << "          eps_abs = " << qpsettings.eps_abs
            << " eps_rel = " << qpsettings.eps_rel << std::endl;
  std::cout << "          eps_prim_inf = " << qpsettings.eps_primal_inf
            << ", eps_dual_inf = " << qpsettings.eps_dual_inf << ","
            << std::endl;

  std::cout << "          rho = " << qpresults.info.rho
            << ", mu_eq = " << qpresults.info.mu_eq
            << ", mu_in = " << qpresults.info.mu_in << "," << std::endl;
  std::cout << "          max_iter = " << qpsettings.max_iter
            << ", max_iter_in = " << qpsettings.max_iter_in << "," << std::endl;
  if (box_constraints) {
    std::cout << "          box constraints: on, " << std::endl;
  } else {
    std::cout << "          box constraints: off, " << std::endl;
  }
  switch (qpsettings.check_solved_option) {
    case CheckSolvedStatus::ITERATION_BASED: {
      std::cout << "          check_solved_option: iteration based, "
                << std::endl
                << "          frequence_infeasibility_check = "
                << qpsettings.frequence_infeasibility_check << std::endl;
      break;
    }
    case CheckSolvedStatus::INTERVAL_BASED: {
      std::cout << "          check_solved_option: interval based, "
                << std::endl
                << "          check_termination = "
                << qpsettings.check_termination << std::endl;
      break;
    }
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
  if (qpsettings.compute_preconditioner) {
    std::cout << "          scaling: on, " << std::endl;
  } else {
    std::cout << "          scaling: off, " << std::endl;
  }
  if (qpsettings.compute_timings) {
    std::cout << "          timings: on, " << std::endl;
  } else {
    std::cout << "          timings: off, " << std::endl;
  }
  switch (qpsettings.initial_guess) {
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
    case QPSolver::PROXQP: {
      break;
    }
    case QPSolver::OSQP: {
      if (qpsettings.adaptive_mu) {
        std::cout << "          adaptive_mu: on, " << std::endl;
        std::cout << "          adaptive_mu_interval: "
                  << qpsettings.adaptive_mu_interval << ", " << std::endl;
        std::cout << "          adaptive_mu_tolerance: "
                  << qpsettings.adaptive_mu_tolerance << ". \n"
                  << std::endl;
      } else {
        std::cout << "          adaptive_mu: off. \n" << std::endl;
      }
      if (qpsettings.polish) {
        std::cout << "          polish: on, " << std::endl;
        std::cout << "          delta: " << qpsettings.delta << ", "
                  << std::endl;
        std::cout << "          polish_refine_iter: "
                  << qpsettings.polish_refine_iter << ". \n"
                  << std::endl;
      } else {
        std::cout << "          polish: off. \n" << std::endl;
      }
      break;
    }
  }
}

template<typename T>
void
print_iteration_line( //
  Results<T>& qpresults,
  const Model<T>& qpmodel,
  const bool box_constraints,
  preconditioner::RuizEquilibration<T>& ruiz,
  const QPSolver solver,
  const isize iter)
{
  ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
  ruiz.unscale_dual_in_place_eq(VectorViewMut<T>{ from_eigen, qpresults.y });
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
  switch (solver) {
    case QPSolver::PROXQP: {
      std::cout << "\033[1;32m[outer iteration " << iter + 1 << "]\033[0m"
                << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | mu_in=" << qpresults.info.mu_in
                << " | rho=" << qpresults.info.rho << std::endl;
      break;
    }
    case QPSolver::OSQP: {
      std::cout << "\033[1;32m[iteration " << iter + 1 << "]\033[0m"
                << std::endl;
      std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << "| primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | mu_eq=" << qpresults.info.mu_eq
                << " | mu_in=" << qpresults.info.mu_in << std::endl;
      break;
    }
  }
  ruiz.scale_primal_in_place(VectorViewMut<T>{ from_eigen, qpresults.x });
  ruiz.scale_dual_in_place_eq(VectorViewMut<T>{ from_eigen, qpresults.y });
  ruiz.scale_dual_in_place_in(
    VectorViewMut<T>{ from_eigen, qpresults.z.head(qpmodel.n_in) });
  if (box_constraints) {
    ruiz.scale_box_dual_in_place_in(
      VectorViewMut<T>{ from_eigen, qpresults.z.tail(qpmodel.dim) });
  }
}

template<typename T>
void
print_solver_statistics( //
  const Settings<T>& qpsettings,
  const Results<T>& qpresults,
  const QPSolver solver)
{
  std::cout << "-------------------SOLVER STATISTICS-------------------"
            << std::endl;
  switch (solver) {
    case QPSolver::PROXQP: {
      std::cout << "outer iter:     " << qpresults.info.iter_ext << std::endl;
      std::cout << "total iter:     " << qpresults.info.iter << std::endl;
      std::cout << "mu updates:     " << qpresults.info.mu_updates << std::endl;
      std::cout << "rho updates:    " << qpresults.info.rho_updates
                << std::endl;
      std::cout << "objective:      " << qpresults.info.objValue << std::endl;
      break;
    }
    case QPSolver::OSQP: {
      std::cout << "total iter:     " << qpresults.info.iter << std::endl;
      std::cout << "mu updates:     " << qpresults.info.mu_updates << std::endl;
      std::cout << "objective:      " << qpresults.info.objValue << std::endl;
      break;
    }
  }
  switch (qpresults.info.status) {
    case QPSolverOutput::QPSOLVER_SOLVED: {
      std::cout << "status:         "
                << "Solved" << std::endl;
      break;
    }
    case QPSolverOutput::QPSOLVER_MAX_ITER_REACHED: {
      std::cout << "status:         "
                << "Maximum number of iterations reached" << std::endl;
      break;
    }
    case QPSolverOutput::QPSOLVER_PRIMAL_INFEASIBLE: {
      std::cout << "status:         "
                << "Primal infeasible" << std::endl;
      break;
    }
    case QPSolverOutput::QPSOLVER_DUAL_INFEASIBLE: {
      std::cout << "status:         "
                << "Dual infeasible" << std::endl;
      break;
    }
    case QPSolverOutput::QPSOLVER_SOLVED_CLOSEST_PRIMAL_FEASIBLE: {
      std::cout << "status:         "
                << "Solved closest primal feasible" << std::endl;
      break;
    }
    case QPSolverOutput::QPSOLVER_NOT_RUN: {
      std::cout << "status:         "
                << "Solver not run" << std::endl;
      break;
    }
  }
  switch (solver) {
    case QPSolver::PROXQP: {
      break;
    }
    case QPSolver::OSQP: {
      if (qpsettings.polish == true) {
        switch (qpresults.info.status_polish) {
          case PolishOutput::POLISH_SUCCEEDED: {
            std::cout << "status_polish:  "
                      << "Success" << std::endl;
            break;
          }
          case PolishOutput::POLISH_FAILED: {
            std::cout << "status_polish:  "
                      << "Failed" << std::endl;
            break;
          }
          case PolishOutput::POLISH_NO_ACTIVE_SET_FOUND: {
            std::cout << "status_polish:  "
                      << "No active set found" << std::endl;
            break;
          }
          case PolishOutput::POLISH_NOT_RUN: {
            std::cout << "status_polish:  "
                      << "Not" << std::endl;
            break;
          }
        }
      }
      break;
    }
  }
  if (qpsettings.compute_timings)
    std::cout << "run time [μs]:  " << qpresults.info.solve_time << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;
}

} // namespace dense
} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_COMMON_DENSE_PRINTS_HPP */
