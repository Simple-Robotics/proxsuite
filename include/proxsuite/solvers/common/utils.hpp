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
#include <proxsuite/proxqp/utils/prints.hpp>
#include <proxsuite/osqp/utils/prints.hpp>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

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
print_setup_header(const pp::Settings<T>& settings,
                   const pp::Results<T>& results,
                   const ppd::Model<T>& model,
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
  switch (qp_solver) {
    case common::QPSolver::PROXQP:
      std::cout << "          max_iter = " << settings.max_iter
                << ", max_iter_in = " << settings.max_iter_in << ","
                << std::endl;
      break;
    case common::QPSolver::OSQP:
      std::cout << "          max_iter = " << settings.max_iter << std::endl;
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
 * Setups the solver.
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

} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_SOLVERS_COMMON_UTILS_HPP */