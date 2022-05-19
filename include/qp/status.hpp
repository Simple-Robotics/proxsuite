/**
 * @file constants.hpp 
*/
#ifndef PROXSUITE_QP_CONSTANTS_HPP
#define PROXSUITE_QP_CONSTANTS_HPP

#include <veg/type_traits/core.hpp>
#include "qp/sparse/fwd.hpp"

namespace proxsuite {
namespace qp {

// STATUS CONSTANTS
enum struct QPSolverOutput{
	PROXQP_SOLVED,
	PROXQP_MAX_ITER_REACHED,
	PROXQP_PRIMAL_INFEASIBLE,
    PROXQP_DUAL_INFEASIBLE
};

} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_CONSTANTS_HPP */
