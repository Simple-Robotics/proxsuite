//
// Copyright (c) 2022 INRIA
//
/**
 * @file wrapper.hpp
 */

#ifndef PROXSUITE_SOLVERS_COMMON_UTILS_HPP
#define PROXSUITE_SOLVERS_COMMON_UTILS_HPP

#include "proxsuite/proxqp/dense/views.hpp"
#include "proxsuite/proxqp/dense/linesearch.hpp"
#include "proxsuite/proxqp/dense/helpers.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include <proxsuite/proxqp/utils/prints.hpp>
#include <proxsuite/osqp/utils/prints.hpp>
#include <cmath>
#include <Eigen/Sparse>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>
#include <proxsuite/linalg/dense/ldlt.hpp>
#include <iomanip>

namespace proxsuite {
namespace solvers {
namespace utils {

// SOLVER TYPE
enum class SolverType {
    PROXQP,
    OSQP
};
/*!
 * Prints the setup header.
 *
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param solver_type PROXQP or OSQP.
 */
template<typename T>
void
print_setup_header(const proxqp::Settings<T>& settings,
                   const proxqp::Results<T>& results,
                   const proxqp::dense::Model<T>& model,
                   const bool box_constraints,
                   const proxqp::DenseBackend& dense_backend,
                   const proxqp::HessianType& hessian_type,
                   const SolverType solver_type)
{

  switch (solver_type) {
    case SolverType::PROXQP:
      proxsuite::proxqp::print_preambule();
      break;
    case SolverType::OSQP:
      proxsuite::osqp::print_preambule();
      break;
  }

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
    case proxqp::DenseBackend::PrimalDualLDLT:
      std::cout << "          dense backend: PrimalDualLDLT, " << std::endl;
      break;
    case proxqp::DenseBackend::PrimalLDLT:
      std::cout << "          dense backend: PrimalLDLT, " << std::endl;
      break;
    case proxqp::DenseBackend::Automatic:
      break;
  }
  switch (hessian_type) {
    case proxqp::HessianType::Dense:
      std::cout << "          problem type: Quadratic Program, " << std::endl;
      break;
    case proxqp::HessianType::Zero:
      std::cout << "          problem type: Linear Program, " << std::endl;
      break;
    case proxqp::HessianType::Diagonal:
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
    case proxqp::InitialGuessStatus::WARM_START:
      std::cout << "          initial guess: warm start. \n" << std::endl;
      break;
    case proxqp::InitialGuessStatus::NO_INITIAL_GUESS:
      std::cout << "          initial guess: no initial guess. \n" << std::endl;
      break;
    case proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:
      std::cout
        << "          initial guess: warm start with previous result. \n"
        << std::endl;
      break;
    case proxqp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:
      std::cout
        << "          initial guess: cold start with previous result. \n"
        << std::endl;
      break;
    case proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:
      std::cout
        << "          initial guess: equality constrained initial guess. \n"
        << std::endl;
  }
}
/*!
 * Setups the solver.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param solver_type PROXQP or OSQP.
 */
template<typename T>
void
setup_solver(
  const proxqp::Settings<T>& qpsettings,
  const proxqp::dense::Model<T>& qpmodel,
  proxqp::Results<T>& qpresults,
  proxqp::dense::Workspace<T>& qpwork,
  const bool box_constraints,
  const proxqp::DenseBackend& dense_backend,
  const proxqp::HessianType& hessian_type,
  proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
  SolverType solver_type)
{

  using namespace proxsuite::proxqp;

  isize n_constraints(qpmodel.n_in);

  if (box_constraints) {
    n_constraints += qpmodel.dim;
  }
  if (qpsettings.compute_timings) {
    qpwork.timer.stop();
    qpwork.timer.start();
  }
  if (qpsettings.verbose) {
    print_setup_header(qpsettings,
                       qpresults,
                       qpmodel,
                       box_constraints,
                       dense_backend,
                       hessian_type,
                       solver_type);
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
        if (solver_type == SolverType::PROXQP) {
          proxqp::dense::linesearch::active_set_change(
            qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        }
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
        if (solver_type == SolverType::PROXQP) {
          proxqp::dense::linesearch::active_set_change(
            qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        }
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
        proxsuite::proxqp::dense::compute_equality_constrained_initial_guess(qpwork,
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
        proxsuite::proxqp::dense::setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        qpwork.n_c = 0;
        for (isize i = 0; i < n_constraints; i++) {
          if (qpresults.z[i] != 0) {
            qpwork.active_inequalities[i] = true;
          } else {
            qpwork.active_inequalities[i] = false;
          }
        }
        if (solver_type == SolverType::PROXQP) {
          proxqp::dense::linesearch::active_set_change(
            qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        }
        break;
      }
      case InitialGuessStatus::NO_INITIAL_GUESS: {
        proxsuite::proxqp::dense::setup_factorization(
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
        proxsuite::proxqp::dense::setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        qpwork.n_c = 0;
        for (isize i = 0; i < n_constraints; i++) {
          if (qpresults.z[i] != 0) {
            qpwork.active_inequalities[i] = true;
          } else {
            qpwork.active_inequalities[i] = false;
          }
        }
        if (solver_type == SolverType::PROXQP) {
          proxqp::dense::linesearch::active_set_change(
            qpmodel, qpresults, dense_backend, n_constraints, qpwork);
        }
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
          proxsuite::proxqp::dense::setup_factorization(
            qpwork, qpmodel, qpresults, dense_backend, hessian_type);
          qpwork.n_c = 0;
          for (isize i = 0; i < n_constraints; i++) {
            if (qpresults.z[i] != 0) {
              qpwork.active_inequalities[i] = true;
            } else {
              qpwork.active_inequalities[i] = false;
            }
          }
          if (solver_type == SolverType::PROXQP) {
            proxqp::dense::linesearch::active_set_change(
              qpmodel, qpresults, dense_backend, n_constraints, qpwork);
          }
          break;
        }
      }
    }
  }
}
/*!
 * Unscales the data at the end of the solve.
 *
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 */
template<typename T>
void
unscale_solver(
  const proxqp::Settings<T>& qpsettings,
  const proxqp::dense::Model<T>& qpmodel,
  proxqp::Results<T>& qpresults,
  const bool box_constraints,
  proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz)
{

  using namespace proxsuite::proxqp;

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
}
/*!
 * Computes the objective function.
 *
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 */
template<typename T>
void
compute_objective(
  const proxqp::dense::Model<T>& qpmodel,
  proxqp::Results<T>& qpresults)
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
/*!
 * Compute timings at the end of the solve.
 *
 * @param qpwork solver workspace.
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 */
template<typename T>
void
compute_timings(
  const proxqp::Settings<T>& qpsettings,
  proxqp::Results<T>& qpresults,
  proxqp::dense::Workspace<T>& qpwork)
{
  if (qpsettings.compute_timings) {
    qpresults.info.solve_time = qpwork.timer.elapsed().user; // in nanoseconds
    qpresults.info.run_time =
      qpresults.info.solve_time + qpresults.info.setup_time;
  }
}
/*!
 * Prints solver statistics after the solve.
 *
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param solver_type PROXQP or OSQP.
 */
template<typename T>
void
print_solver_statistics(
  const proxqp::Settings<T>& qpsettings,
  proxqp::Results<T>& qpresults,
  SolverType solver_type)
{
  using namespace proxsuite::proxqp;

  if (qpsettings.verbose) {

    std::cout << "-------------------SOLVER STATISTICS-------------------"
                << std::endl;

    switch (solver_type) {
      case SolverType::PROXQP: {
        std::cout << "outer iter:   " << qpresults.info.iter_ext << std::endl;
        std::cout << "total iter:   " << qpresults.info.iter << std::endl;
        std::cout << "mu updates:   " << qpresults.info.mu_updates << std::endl;
        std::cout << "rho updates:  " << qpresults.info.rho_updates << std::endl;
        std::cout << "objective:    " << qpresults.info.objValue << std::endl;
        break;
      case SolverType::OSQP: {
        std::cout << "iter:   " << qpresults.info.iter_ext << std::endl;
        std::cout << "objective:    " << qpresults.info.objValue << std::endl;
        break;
      }
      }
    }
    
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
}
/*!
 * Clean work data to prepare the next solve.
 *
 * @param qpwork solver workspace.
 * @param qpresults solver results.
 */
template<typename T>
void
prepare_next_solve(
  proxqp::Results<T>& qpresults,
  proxqp::dense::Workspace<T>& qpwork)
{

  qpwork.dirty = true;
  qpwork.is_initialized = true; // necessary because we call workspace cleanup

  assert(!std::isnan(qpresults.info.pri_res));
  assert(!std::isnan(qpresults.info.dua_res));
  assert(!std::isnan(qpresults.info.duality_gap));
}
/*!
 * Computes residuals and infeasibility. 
 * Breaks if problem primal infeasible and we do not want to solve the closest, or if the problem is solved.
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
compute_residuals_and_infeasibility_1( //
  const proxqp::Settings<T>& qpsettings,
  const proxqp::dense::Model<T>& qpmodel,
  proxqp::Results<T>& qpresults,
  proxqp::dense::Workspace<T>& qpwork,
  const bool box_constraints,
  const proxqp::HessianType& hessian_type,
  proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
  SolverType solver_type,
  T& primal_feasibility_eq_rhs_0,
  T& primal_feasibility_in_rhs_0,
  T& primal_feasibility_eq_lhs,
  T& primal_feasibility_in_lhs,
  T& primal_feasibility_lhs,
  T& dual_feasibility_lhs,
  T& dual_feasibility_rhs_0,
  T& dual_feasibility_rhs_1,
  T& dual_feasibility_rhs_3,
  T& rhs_duality_gap,
  T& duality_gap,
  T& scaled_eps,
  proxqp::i64 iter)
{

    using namespace proxsuite::proxqp;

    dense::global_primal_residual(qpmodel,
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

    dense::global_dual_residual(qpresults,
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
      
      proxsuite::solvers::utils::compute_objective(qpmodel, qpresults);
      
      std::cout << "\033[1;32m[outer iteration " << iter + 1 << "]\033[0m"
                << std::endl;

      switch (solver_type) {
        case SolverType::PROXQP: {
          std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << " | primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | mu_in=" << qpresults.info.mu_in
                << " | rho=" << qpresults.info.rho << std::endl;
          break;
        case SolverType::OSQP: {
          std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                << " | primal residual=" << qpresults.info.pri_res
                << " | dual residual=" << qpresults.info.dua_res
                << " | duality gap=" << qpresults.info.duality_gap
                << " | rho=" << qpresults.info.rho
                << " | mu_eq=" << qpresults.info.mu_eq << std::endl;
          break;
        }
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
          return true; // The loop should stop
        }
      } else {
        qpresults.info.status = QPSolverOutput::PROXQP_SOLVED;
        return true; // The loop should stop
      }
    }

    return false; // Should not break the loop
}
/*!
 * Computes residuals and infeasibility. 
 * Breaks if problem primal infeasible and we do not want to solve the closest, or if the problem is solved.
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
compute_residuals_and_infeasibility_2( //
  const proxqp::Settings<T>& qpsettings,
  const proxqp::dense::Model<T>& qpmodel,
  proxqp::Results<T>& qpresults,
  proxqp::dense::Workspace<T>& qpwork,
  const bool box_constraints,
  const proxqp::HessianType& hessian_type,
  proxqp::dense::preconditioner::RuizEquilibration<T>& ruiz,
  T& primal_feasibility_eq_rhs_0,
  T& primal_feasibility_in_rhs_0,
  T& primal_feasibility_eq_lhs,
  T& primal_feasibility_in_lhs,
  T& primal_feasibility_lhs_new,
  T& dual_feasibility_lhs,
  T& dual_feasibility_rhs_0,
  T& dual_feasibility_rhs_1,
  T& dual_feasibility_rhs_3,
  T& rhs_duality_gap,
  T& duality_gap,
  T& scaled_eps)
{

    using namespace proxsuite::proxqp;

    dense::global_primal_residual(qpmodel,
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

    bool is_primal_feasible =
      primal_feasibility_lhs_new <=
      (scaled_eps + qpsettings.eps_rel * std::max(primal_feasibility_eq_rhs_0,
                                                  primal_feasibility_in_rhs_0));
    qpresults.info.pri_res = primal_feasibility_lhs_new;
    if (is_primal_feasible) {
      T dual_feasibility_lhs_new(dual_feasibility_lhs);

      dense::global_dual_residual(qpresults,
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

      bool is_dual_feasible =
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
                  proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
              qpresults.info.status =
                proxqp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
            } else {
              qpresults.info.status = proxqp::QPSolverOutput::PROXQP_SOLVED;
            }
          }
        } else {
          if (qpsettings.primal_infeasibility_solving &&
              qpresults.info.status ==
                proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
            qpresults.info.status =
              proxqp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
          } else {
            qpresults.info.status = proxqp::QPSolverOutput::PROXQP_SOLVED;
          }
        }
      }
    }

}

} // namespace utils
} // namespace solvers
} // namespace proxsuite


#endif /* end of include guard PROXSUITE_SOLVERS_COMMON_UTILS_HPP */