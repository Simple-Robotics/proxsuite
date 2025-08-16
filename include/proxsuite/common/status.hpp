//
// Copyright (c) 2022 INRIA
//
/**
 * @file constants.hpp
 */
#ifndef PROXSUITE_COMMON_STATUS_HPP
#define PROXSUITE_COMMON_STATUS_HPP

namespace proxsuite {
namespace common {

// SOLVER STATUS
enum struct QPSolverOutput
{
  QPSOLVER_SOLVED,            // the problem is solved.
  QPSOLVER_MAX_ITER_REACHED,  // the maximum number of iterations has been
                              // reached.
  QPSOLVER_PRIMAL_INFEASIBLE, // the problem is primal infeasible.
  QPSOLVER_SOLVED_CLOSEST_PRIMAL_FEASIBLE, // the closest (in L2 sense) feasible
                                           // problem is solved.
  QPSOLVER_DUAL_INFEASIBLE,                // the problem is dual infeasible.
  QPSOLVER_NOT_RUN                         // the solver has not been run yet.
};
// INITIAL GUESS STATUS
enum struct InitialGuessStatus
{
  NO_INITIAL_GUESS,
  EQUALITY_CONSTRAINED_INITIAL_GUESS,
  WARM_START_WITH_PREVIOUS_RESULT,
  WARM_START,
  COLD_START_WITH_PREVIOUS_RESULT
};
// PRECONDITIONER STATUS
enum struct PreconditionerStatus
{
  EXECUTE, // initialize or update with qp in entry
  KEEP,    // keep previous preconditioner (for update method)
  IDENTITY // do not execute, hence use identity preconditioner (for init
           // method)
};
// POLISH (OSQP) STATUS
enum struct PolishStatus
{
  POLISH_FAILED,             // polishing failed.
  POLISH_NOT_RUN,            // polishing have not been run yet.
  POLISH_SUCCEEDED,          // residuals are reduced.
  POLISH_NO_ACTIVE_SET_FOUND // no active set detected, polishing skipped.
};

} // namespace common
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_COMMON_STATUS_HPP */
