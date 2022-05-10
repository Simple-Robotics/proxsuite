#ifndef PROXSUITE_INCLUDE_QP_CONSTANTS_HPP
#define PROXSUITE_INCLUDE_QP_CONSTANTS_HPP

#include <veg/type_traits/core.hpp>

namespace proxsuite{
namespace qp {
using veg::isize;

// STATUS CONSTANTS
const isize PROXQP_SOLVED = 1;
const isize PROXQP_SOLVED_INACCURATE = 2;
const isize PROXQP_MAX_ITER_REACHED = -2;
const isize PROXQP_PRIMAL_INFEASIBLE = -3;
const isize PROXQP_PRIMAL_INFEASIBLE_INACCURATE = 3;
const isize PROXQP_DUAL_INFEASIBLE = -4;
const isize PROXQP_DUAL_INFEASIBLE_INACCURATE = 4;
const isize PROXQP_SIGINT = -5;
const isize PROXQP_TIME_LIMIT_REACHED = -6;
const isize PROXQP_UNSOLVED = -10;
const isize PROXQP_NON_CVX = -7;

} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_INCLUDE_QP_CONSTANTS_HPP */