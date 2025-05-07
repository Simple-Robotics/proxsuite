//
// Copyright (c) 2025 INRIA
//
/**
 * @file utils.hpp
 */

#ifndef PROXSUITE_SOLVERS_COMMON_UTILS_HPP
#define PROXSUITE_SOLVERS_COMMON_UTILS_HPP

#include "proxsuite/proxqp/settings.hpp"
#include "proxsuite/proxqp/results.hpp"
#include "proxsuite/proxqp/dense/model.hpp"
#include "proxsuite/proxqp/dense/workspace.hpp"
#include "proxsuite/proxqp/dense/preconditioner/ruiz.hpp"
#include "proxsuite/proxqp/dense/helpers.hpp"
#include "proxsuite/proxqp/dense/linesearch.hpp"
#include "proxsuite/proxqp/dense/utils.hpp"
#include <proxsuite/proxqp/utils/prints.hpp>
#include <proxsuite/osqp/utils/prints.hpp>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>
#include <iomanip>

namespace proxsuite {
namespace common {

namespace pp = proxsuite::proxqp;
namespace ppd = proxsuite::proxqp::dense;
namespace ppdp = proxsuite::proxqp::dense::preconditioner;
namespace plv = proxsuite::linalg::veg;

///
/// @brief This enum defines the different solvers implemented in ProxSuite.
///
enum class QPSolver
{
  PROXQP,
  OSQP
};
/*!
 * Prints the setup header.
 *
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param qp_solver PROXQP or OSQP.
 */
template<typename T>
void
print_setup_header(const pp::Settings<T>& qpsettings,
                   const pp::Results<T>& qpresults,
                   const ppd::Model<T>& qpmodel,
                   const bool box_constraints,
                   const pp::DenseBackend& dense_backend,
                   const pp::HessianType& hessian_type,
                   const common::QPSolver qp_solver)
{

  switch (qp_solver) {
    case common::QPSolver::PROXQP:
      proxsuite::proxqp::print_preambule();
      break;
    case common::QPSolver::OSQP:
      proxsuite::osqp::print_preambule();
      break;
  }

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
  switch (qp_solver) {
    case common::QPSolver::PROXQP:
      std::cout << "          max_iter = " << qpsettings.max_iter
                << ", max_iter_in = " << qpsettings.max_iter_in << ","
                << std::endl;
      break;
    case common::QPSolver::OSQP:
      std::cout << "          max_iter = " << qpsettings.max_iter << std::endl;
      break;
  }
  if (box_constraints) {
    std::cout << "          box constraints: on, " << std::endl;
  } else {
    std::cout << "          box constraints: off, " << std::endl;
  }
  switch (dense_backend) {
    case pp::DenseBackend::PrimalDualLDLT:
      std::cout << "          dense backend: PrimalDualLDLT, " << std::endl;
      break;
    case pp::DenseBackend::PrimalLDLT:
      std::cout << "          dense backend: PrimalLDLT, " << std::endl;
      break;
    case pp::DenseBackend::Automatic:
      break;
  }
  switch (hessian_type) {
    case pp::HessianType::Dense:
      std::cout << "          problem type: Quadratic Program, " << std::endl;
      break;
    case pp::HessianType::Zero:
      std::cout << "          problem type: Linear Program, " << std::endl;
      break;
    case pp::HessianType::Diagonal:
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
    case pp::InitialGuessStatus::WARM_START:
      std::cout << "          initial guess: warm start. \n" << std::endl;
      break;
    case pp::InitialGuessStatus::NO_INITIAL_GUESS:
      std::cout << "          initial guess: no initial guess. \n" << std::endl;
      break;
    case pp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:
      std::cout
        << "          initial guess: warm start with previous result. \n"
        << std::endl;
      break;
    case pp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:
      std::cout
        << "          initial guess: cold start with previous result. \n"
        << std::endl;
      break;
    case pp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:
      std::cout
        << "          initial guess: equality constrained initial guess. \n"
        << std::endl;
  }
}
/*!
 * Prints the solver's statistics.
 *
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param qp_solver PROXQP or OSQP.
 */
template<typename T>
void
print_solver_statistics(const pp::Settings<T>& qpsettings,
                        const pp::Results<T>& qpresults,
                        const common::QPSolver qp_solver)
{
  std::cout << "-------------------SOLVER STATISTICS-------------------"
            << std::endl;

  switch (qp_solver) {
    case common::QPSolver::PROXQP: {
      std::cout << "outer iter:     " << qpresults.info.iter_ext << std::endl;
      std::cout << "total iter:     " << qpresults.info.iter << std::endl;
      std::cout << "mu updates:     " << qpresults.info.mu_updates << std::endl;
      std::cout << "rho updates:    " << qpresults.info.rho_updates
                << std::endl;
      std::cout << "objective:      " << qpresults.info.objValue << std::endl;
      break;
    }
    case common::QPSolver::OSQP: {
      std::cout << "outer iter:     " << qpresults.info.iter_ext << std::endl;
      std::cout << "total iter:     " << qpresults.info.iter_ext << std::endl;
      std::cout << "mu updates:     " << qpresults.info.mu_updates << std::endl;
      std::cout << "objective:      " << qpresults.info.objValue << std::endl;
      break;
    }
  }

  switch (qpresults.info.status) {
    case pp::QPSolverOutput::PROXQP_SOLVED: {
      std::cout << "status:         "
                << "Solved" << std::endl;
      break;
    }
    case pp::QPSolverOutput::PROXQP_MAX_ITER_REACHED: {
      std::cout << "status:         "
                << "Maximum number of iterations reached" << std::endl;
      break;
    }
    case pp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE: {
      std::cout << "status:         "
                << "Primal infeasible" << std::endl;
      break;
    }
    case pp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE: {
      std::cout << "status:         "
                << "Dual infeasible" << std::endl;
      break;
    }
    case pp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: {
      std::cout << "status:         "
                << "Solved closest primal feasible" << std::endl;
      break;
    }
    case pp::QPSolverOutput::PROXQP_NOT_RUN: {
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
/*!
 * Prepares the next solve. Sets workspace to initialized and
 * cleanups the information results.
 *
 * @param qpwork solver workspace.
 * @param qpresults solver results.
 */
template<typename T>
void
prepare_next_solve(pp::Results<T>& qpresults, ppd::Workspace<T>& qpwork)
{
  qpwork.dirty = true;
  qpwork.is_initialized = true; // necessary because we call workspace cleanup

  assert(!std::isnan(qpresults.info.pri_res));
  assert(!std::isnan(qpresults.info.dua_res));
  assert(!std::isnan(qpresults.info.duality_gap));
}
/*!
 * Setups the solver.
 * In particular, it scales (Ruiz equilibration) the data, then
 * builds the KKT matrix according to the algorihm, eg:
 * proxqp: Builds the KKT with equality and activate inequality constraints
 * osqp: Builds the KKT with all of the constraints.
 *
 * @param qpwork solver workspace.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpsettings solver settings.
 * @param qpresults solver results.
 * @param ruiz ruiz preconditioner.
 * @param qp_solver PROXQP or OSQP.
 */
template<typename T>
void
setup_solver(const pp::Settings<T>& qpsettings,
             const ppd::Model<T>& qpmodel,
             pp::Results<T>& qpresults,
             ppd::Workspace<T>& qpwork,
             const bool box_constraints,
             const pp::DenseBackend& dense_backend,
             const pp::HessianType& hessian_type,
             ppdp::RuizEquilibration<T>& ruiz,
             QPSolver qp_solver)
{
  plv::isize n_constraints(qpmodel.n_in);
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
                       qp_solver);
  }
  if (qpwork.dirty) { // the following is used when a solve has already been
                      // executed (and without any intermediary model update)
    switch (qpsettings.initial_guess) {
      case pp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
        qpwork.cleanup(box_constraints);
        qpresults.cleanup(qpsettings);
        break;
      }
      case pp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
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
      case pp::InitialGuessStatus::NO_INITIAL_GUESS: {
        qpwork.cleanup(box_constraints);
        qpresults.cleanup(qpsettings);
        break;
      }
      case pp::InitialGuessStatus::WARM_START: {
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
      case pp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
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
        pp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT) {
      switch (hessian_type) {
        case pp::HessianType::Zero:
          break;
        case pp::HessianType::Dense:
          qpwork.H_scaled = qpmodel.H;
          break;
        case pp::HessianType::Diagonal:
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
      case pp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
        compute_equality_constrained_initial_guess(qpwork,
                                                   qpsettings,
                                                   qpmodel,
                                                   n_constraints,
                                                   dense_backend,
                                                   hessian_type,
                                                   qpresults);
        break;
      }
      case pp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
        switch (qp_solver) {
          case common::QPSolver::PROXQP: {
            //!\ TODO in a quicker way
            qpwork.n_c = 0;
            for (plv::isize i = 0; i < n_constraints; i++) {
              if (qpresults.z[i] != 0) {
                qpwork.active_inequalities[i] = true;
              } else {
                qpwork.active_inequalities[i] = false;
              }
            }
            ppd::linesearch::active_set_change(
              qpmodel, qpresults, dense_backend, n_constraints, qpwork);
          } break;
          case common::QPSolver::OSQP: {
            // TODO: Call for function to build the full KKT
          } break;
        }
        break;
      }
      case pp::InitialGuessStatus::NO_INITIAL_GUESS: {
        break;
      }
      case pp::InitialGuessStatus::WARM_START: {
        switch (qp_solver) {
          case common::QPSolver::PROXQP: {
            //!\ TODO in a quicker way
            qpwork.n_c = 0;
            for (plv::isize i = 0; i < n_constraints; i++) {
              if (qpresults.z[i] != 0) {
                qpwork.active_inequalities[i] = true;
              } else {
                qpwork.active_inequalities[i] = false;
              }
            }
            ppd::linesearch::active_set_change(
              qpmodel, qpresults, dense_backend, n_constraints, qpwork);
          } break;
          case common::QPSolver::OSQP: {
            // TODO: Call for function to build the full KKT
          } break;
        }
        break;
      }
      case pp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
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
      case pp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
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
      case pp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
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
        switch (qp_solver) {
          case common::QPSolver::PROXQP: {
            //!\ TODO in a quicker way
            qpwork.n_c = 0;
            for (plv::isize i = 0; i < n_constraints; i++) {
              if (qpresults.z[i] != 0) {
                qpwork.active_inequalities[i] = true;
              } else {
                qpwork.active_inequalities[i] = false;
              }
            }
            ppd::linesearch::active_set_change(
              qpmodel, qpresults, dense_backend, n_constraints, qpwork);
          } break;
          case common::QPSolver::OSQP: {
            // TODO: Call for function to build the full KKT
          } break;
        }
        break;
      }
      case pp::InitialGuessStatus::NO_INITIAL_GUESS: {
        setup_factorization(
          qpwork, qpmodel, qpresults, dense_backend, hessian_type);
        break;
      }
      case pp::InitialGuessStatus::WARM_START: {
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
        switch (qp_solver) {
          case common::QPSolver::PROXQP: {
            //!\ TODO in a quicker way
            qpwork.n_c = 0;
            for (plv::isize i = 0; i < n_constraints; i++) {
              if (qpresults.z[i] != 0) {
                qpwork.active_inequalities[i] = true;
              } else {
                qpwork.active_inequalities[i] = false;
              }
            }
            ppd::linesearch::active_set_change(
              qpmodel, qpresults, dense_backend, n_constraints, qpwork);
          } break;
          case common::QPSolver::OSQP: {
            // TODO: Call for function to build the full KKT
          } break;
        }
        break;
      }
      case pp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
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
          switch (qp_solver) {
            case common::QPSolver::PROXQP: {
              //!\ TODO in a quicker way
              qpwork.n_c = 0;
              for (plv::isize i = 0; i < n_constraints; i++) {
                if (qpresults.z[i] != 0) {
                  qpwork.active_inequalities[i] = true;
                } else {
                  qpwork.active_inequalities[i] = false;
                }
              }
              ppd::linesearch::active_set_change(
                qpmodel, qpresults, dense_backend, n_constraints, qpwork);
            } break;
            case common::QPSolver::OSQP: {
              // TODO: Call for function to build the full KKT
            } break;
          }
          break;
        }
      }
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
compute_objective(const ppd::Model<T>& qpmodel, pp::Results<T>& qpresults)
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
/*!
 * Computes the residuals and the feasibility of the problem, then update it
 * and stops the algorithm if needed.
 *
 * @param qpsettings solver settings.
 * @param qpmodel QP problem model as defined by the user (without any scaling
 * performed).
 * @param qpresults solver results.
 * @param qpwork solver workspace.
 * @param ruiz ruiz preconditioner.
 * @param qp_solver PROXQP or OSQP.
 */
template<typename T>
void
compute_feasibility( //
  const pp::Settings<T>& qpsettings,
  const ppd::Model<T>& qpmodel,
  pp::Results<T>& qpresults,
  ppd::Workspace<T>& qpwork,
  const bool box_constraints,
  const pp::HessianType& hessian_type,
  ppdp::RuizEquilibration<T>& ruiz,
  QPSolver qp_solver,
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
  plv::i64 iter,
  bool& stop_loop)
{

  ppd::global_primal_residual(qpmodel,
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

  ppd::global_dual_residual(qpresults,
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
      std::max(std::max(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
               std::max(dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2));
  }
  bool is_dual_feasible = dual_feasibility_lhs <= rhs_dua;

  if (qpsettings.verbose) {
    ruiz.unscale_primal_in_place(
      pp::VectorViewMut<T>{ pp::from_eigen, qpresults.x });
    ruiz.unscale_dual_in_place_eq(
      pp::VectorViewMut<T>{ pp::from_eigen, qpresults.y });
    ruiz.unscale_dual_in_place_in(
      pp::VectorViewMut<T>{ pp::from_eigen, qpresults.z.head(qpmodel.n_in) });
    if (box_constraints) {
      ruiz.unscale_box_dual_in_place_in(
        pp::VectorViewMut<T>{ pp::from_eigen, qpresults.z.tail(qpmodel.dim) });
    }

    compute_objective(qpmodel, qpresults);

    std::cout << "\033[1;32m[outer iteration " << iter + 1 << "]\033[0m"
              << std::endl;

    switch (qp_solver) {
      case common::QPSolver::PROXQP: {
        std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                  << " | primal residual=" << qpresults.info.pri_res
                  << " | dual residual=" << qpresults.info.dua_res
                  << " | duality gap=" << qpresults.info.duality_gap
                  << " | mu_in=" << qpresults.info.mu_in
                  << " | rho=" << qpresults.info.rho << std::endl;
        break;
        case common::QPSolver::OSQP: {
          std::cout << std::scientific << std::setw(2) << std::setprecision(2)
                    << " | primal residual=" << qpresults.info.pri_res
                    << " | dual residual=" << qpresults.info.dua_res
                    << " | duality gap=" << qpresults.info.duality_gap
                    << " | rho=" << qpresults.info.rho
                    << " | mu_in=" << qpresults.info.mu_in
                    << " | mu_eq=" << qpresults.info.mu_eq << std::endl;
          break;
        }
      }
    }

    ruiz.scale_primal_in_place(
      pp::VectorViewMut<T>{ pp::from_eigen, qpresults.x });
    ruiz.scale_dual_in_place_eq(
      pp::VectorViewMut<T>{ pp::from_eigen, qpresults.y });
    ruiz.scale_dual_in_place_in(
      pp::VectorViewMut<T>{ pp::from_eigen, qpresults.z.head(qpmodel.n_in) });
    if (box_constraints) {
      ruiz.scale_box_dual_in_place_in(
        pp::VectorViewMut<T>{ pp::from_eigen, qpresults.z.tail(qpmodel.dim) });
    }
  }

  if (is_primal_feasible && is_dual_feasible) {
    if (qpsettings.check_duality_gap) {
      if (std::fabs(qpresults.info.duality_gap) <=
          qpsettings.eps_duality_gap_abs +
            qpsettings.eps_duality_gap_rel * rhs_duality_gap) {
        if (qpsettings.primal_infeasibility_solving &&
            qpresults.info.status ==
              pp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
          qpresults.info.status =
            pp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
        } else {
          qpresults.info.status = pp::QPSolverOutput::PROXQP_SOLVED;
        }
        stop_loop = true;
      }
    } else {
      qpresults.info.status = pp::QPSolverOutput::PROXQP_SOLVED;
      stop_loop = true;
    }
  }
}
/*!
 * Computes residuals, the infeasibility and updates the solver's status.
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
update_solver_status( //
  const pp::Settings<T>& qpsettings,
  const ppd::Model<T>& qpmodel,
  pp::Results<T>& qpresults,
  ppd::Workspace<T>& qpwork,
  const bool box_constraints,
  const pp::HessianType& hessian_type,
  ppdp::RuizEquilibration<T>& ruiz,
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
  ppd::global_primal_residual(qpmodel,
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

    ppd::global_dual_residual(qpresults,
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
                pp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
            qpresults.info.status =
              pp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
          } else {
            qpresults.info.status = pp::QPSolverOutput::PROXQP_SOLVED;
          }
        }
      } else {
        if (qpsettings.primal_infeasibility_solving &&
            qpresults.info.status ==
              pp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
          qpresults.info.status =
            pp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE;
        } else {
          qpresults.info.status = pp::QPSolverOutput::PROXQP_SOLVED;
        }
      }
    }
  }
}

} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_SOLVERS_COMMON_UTILS_HPP */