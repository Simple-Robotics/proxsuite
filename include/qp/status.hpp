/**
 * @file constants.hpp 
*/
#ifndef PROXSUITE_QP_CONSTANTS_HPP
#define PROXSUITE_QP_CONSTANTS_HPP

#include <veg/type_traits/core.hpp>
#include "qp/sparse/fwd.hpp"

namespace proxsuite {
namespace qp {

// SOLVER STATUS
enum struct QPSolverOutput {
	PROXQP_SOLVED,
	PROXQP_MAX_ITER_REACHED,
	PROXQP_PRIMAL_INFEASIBLE,
	PROXQP_DUAL_INFEASIBLE
};
// INITIAL GUESS STATUS
enum struct InitialGuessStatus {
	NO_INITIAL_GUESS,
	EQUALITY_CONSTRAINED_INITIAL_GUESS, 
	WARM_START_WITH_PREVIOUS_RESULT,
	WARM_START, // tout refaire pour la KKT avec la partie active de z
	COLD_START_WITH_PREVIOUS_RESULT // keep solutions + KKT
};
// PRECONDITIONER STATUS
enum struct PreconditionerStatus {
	EXECUTE,// initialize or update with qp in entry
	KEEP, // keep previous preconditioner (for update method)
	IDENTITY // do not execute, hence use identity preconditioner (for init method)
};


} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_CONSTANTS_HPP */
