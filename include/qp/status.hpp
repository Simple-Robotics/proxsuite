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
enum struct QPSolverOutput{
	PROXQP_SOLVED,
	PROXQP_MAX_ITER_REACHED,
	PROXQP_PRIMAL_INFEASIBLE,
    PROXQP_DUAL_INFEASIBLE
};
// INITIAL GUESS STATUS
enum struct InitialGuessStatus{
	NO_INITIAL_GUESS,
	UNCONSTRAINED_INITIAL_GUESS,
	EQUALITY_CONSTRAINED_INITIAL_GUESS,
    WARM_START_WITH_PREVIOUS_RESULT,
	WARM_START,
    COLD_START
};

} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_CONSTANTS_HPP */
