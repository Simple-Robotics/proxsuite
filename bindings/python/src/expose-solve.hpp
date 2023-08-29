//
// Copyright (c) 2022 INRIA
//
#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
solveDenseQp(pybind11::module_ m)
{
  m.def(
    "solve",
    pybind11::overload_cast<optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<bool>,
                            bool,
                            bool,
                            optional<isize>,
                            proxsuite::proxqp::InitialGuessStatus,
                            bool,
                            optional<T>,
                            optional<T>,
                            bool,
                            optional<T>>(&dense::solve<T>),
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    pybind11::arg_v("H", nullopt, "quadratic cost with dense format."),
    pybind11::arg_v("g", nullopt, "linear cost"),
    pybind11::arg_v(
      "A", nullopt, "equality constraint matrix with dense format."),
    pybind11::arg_v("b", nullopt, "equality constraint vector"),
    pybind11::arg_v(
      "C", nullopt, "inequality constraint matrix with dense format."),
    pybind11::arg_v("l", nullopt, "lower inequality constraint vector"),
    pybind11::arg_v("u", nullopt, "upper inequality constraint vector"),
    pybind11::arg_v("x", nullopt, "primal warm start"),
    pybind11::arg_v("y", nullopt, "dual equality warm start"),
    pybind11::arg_v("z", nullopt, "dual inequality warm start"),
    pybind11::arg_v(
      "eps_abs",
      nullopt,
      "absolute accuracy level used for the solver stopping criterion."),
    pybind11::arg_v("eps_rel",
                    nullopt,
                    "relative accuracy level used for the solver stopping "
                    "criterion. Deactivated in standard settings."),
    pybind11::arg_v("rho", nullopt, "primal proximal parameter"),
    pybind11::arg_v(
      "mu_eq", nullopt, "dual equality constraint proximal parameter"),
    pybind11::arg_v(
      "mu_in", nullopt, "dual inequality constraint proximal parameter"),
    pybind11::arg_v("verbose",
                    nullopt,
                    "verbose option to print information at each iteration."),
    pybind11::arg_v("compute_preconditioner",
                    true,
                    "executes the default preconditioner for reducing ill "
                    "conditioning and speeding up the solver."),
    pybind11::arg_v("compute_timings", false, "compute solver's timings."),
    pybind11::arg_v("max_iter", nullopt, "maximum number of iteration."),
    pybind11::arg_v(
      "initial_guess",
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      "maximum number of iteration."),
    pybind11::arg_v(
      "check_duality_gap",
      false,
      "if set to true, include the duality gap in absolute and relative "
      "stopping criteria."),
    pybind11::arg_v("eps_duality_gap_abs",
                    nullopt,
                    "absolute accuracy threshold used for the duality-gap "
                    "stopping criterion."),
    pybind11::arg_v("eps_duality_gap_rel",
                    nullopt,
                    "relative accuracy threshold used for the duality-gap "
                    "stopping criterion."),
    pybind11::arg_v("primal_infeasibility_solving",
                    false,
                    "solves the closest feasible problem in L2 sense "
                    "if the QP problem appears to be infeasible."),
    pybind11::arg_v("default_H_eigenvalue_estimate",
                    0.,
                    "Default estimate of the minimal eigen value of H."));

  m.def(
    "solve",
    pybind11::overload_cast<optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::MatRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<dense::VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<VecRef<T>>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<T>,
                            optional<bool>,
                            bool,
                            bool,
                            optional<isize>,
                            proxsuite::proxqp::InitialGuessStatus,
                            bool,
                            optional<T>,
                            optional<T>,
                            bool,
                            optional<T>>(&dense::solve<T>),
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    pybind11::arg_v("H", nullopt, "quadratic cost with dense format."),
    pybind11::arg_v("g", nullopt, "linear cost"),
    pybind11::arg_v(
      "A", nullopt, "equality constraint matrix with dense format."),
    pybind11::arg_v("b", nullopt, "equality constraint vector"),
    pybind11::arg_v(
      "C", nullopt, "inequality constraint matrix with dense format."),
    pybind11::arg_v("l", nullopt, "lower inequality constraint vector"),
    pybind11::arg_v("u", nullopt, "upper inequality constraint vector"),
    pybind11::arg_v("l_box", nullopt, "lower box inequality constraint vector"),
    pybind11::arg_v("u_box", nullopt, "upper box inequality constraint vector"),
    pybind11::arg_v("x", nullopt, "primal warm start"),
    pybind11::arg_v("y", nullopt, "dual equality warm start"),
    pybind11::arg_v("z", nullopt, "dual inequality warm start"),
    pybind11::arg_v(
      "eps_abs",
      nullopt,
      "absolute accuracy level used for the solver stopping criterion."),
    pybind11::arg_v("eps_rel",
                    nullopt,
                    "relative accuracy level used for the solver stopping "
                    "criterion. Deactivated in standard settings."),
    pybind11::arg_v("rho", nullopt, "primal proximal parameter"),
    pybind11::arg_v(
      "mu_eq", nullopt, "dual equality constraint proximal parameter"),
    pybind11::arg_v(
      "mu_in", nullopt, "dual inequality constraint proximal parameter"),
    pybind11::arg_v("verbose",
                    nullopt,
                    "verbose option to print information at each iteration."),
    pybind11::arg_v("compute_preconditioner",
                    true,
                    "executes the default preconditioner for reducing ill "
                    "conditioning and speeding up the solver."),
    pybind11::arg_v("compute_timings", false, "compute solver's timings."),
    pybind11::arg_v("max_iter", nullopt, "maximum number of iteration."),
    pybind11::arg_v(
      "initial_guess",
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      "maximum number of iteration."),
    pybind11::arg_v(
      "check_duality_gap",
      false,
      "if set to true, include the duality gap in absolute and relative "
      "stopping criteria."),
    pybind11::arg_v("eps_duality_gap_abs",
                    nullopt,
                    "absolute accuracy threshold used for the duality-gap "
                    "stopping criterion."),
    pybind11::arg_v("eps_duality_gap_rel",
                    nullopt,
                    "relative accuracy threshold used for the duality-gap "
                    "stopping criterion."),
    pybind11::arg_v("primal_infeasibility_solving",
                    false,
                    "solves the closest feasible problem in L2 sense "
                    "if the QP problem appears to be infeasible."),
    pybind11::arg_v("default_H_eigenvalue_estimate",
                    0.,
                    "Default estimate of the minimal eigen value of H."));
}

} // namespace python
} // namespace dense

namespace sparse {
namespace python {

template<typename T, typename I>
void
solveSparseQp(pybind11::module_ m)
{
  m.def(
    "solve",
    &sparse::solve<T, I>,
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    pybind11::arg_v("H", nullopt, "quadratic cost with sparse format."),
    pybind11::arg_v("g", nullopt, "linear cost"),
    pybind11::arg_v(
      "A", nullopt, "equality constraint matrix with sparse format."),
    pybind11::arg_v("b", nullopt, "equality constraint vector"),
    pybind11::arg_v(
      "C", nullopt, "inequality constraint matrix with sparse format."),
    pybind11::arg_v("l", nullopt, "lower inequality constraint vector"),
    pybind11::arg_v("u", nullopt, "upper inequality constraint vector"),
    pybind11::arg_v("x", nullopt, "primal warm start"),
    pybind11::arg_v("y", nullopt, "dual equality warm start"),
    pybind11::arg_v("z", nullopt, "dual inequality warm start"),
    pybind11::arg_v(
      "eps_abs",
      nullopt,
      "absolute accuracy level used for the solver stopping criterion."),
    pybind11::arg_v("eps_rel",
                    nullopt,
                    "relative accuracy level used for the solver stopping "
                    "criterion. Deactivated in standard settings."),
    pybind11::arg_v("rho", nullopt, "primal proximal parameter"),
    pybind11::arg_v(
      "mu_eq", nullopt, "dual equality constraint proximal parameter"),
    pybind11::arg_v(
      "mu_in", nullopt, "dual inequality constraint proximal parameter"),
    pybind11::arg_v("verbose",
                    nullopt,
                    "verbose option to print information at each iteration."),
    pybind11::arg_v("compute_preconditioner",
                    true,
                    "executes the default preconditioner for reducing ill "
                    "conditioning and speeding up the solver."),
    pybind11::arg_v("compute_timings", false, "compute solver's timings."),
    pybind11::arg_v("max_iter", nullopt, "maximum number of iteration."),
    pybind11::arg_v("initial_guess",
                    proxsuite::proxqp::InitialGuessStatus::
                      EQUALITY_CONSTRAINED_INITIAL_GUESS),
    pybind11::arg_v("sparse_backend",
                    proxsuite::proxqp::SparseBackend::Automatic),
    pybind11::arg_v("check_duality_gap",
                    false,
                    "if set to true, include the duality gap in absolute and "
                    "relative stopping criteria."),
    pybind11::arg_v("eps_duality_gap_abs",
                    nullopt,
                    "absolute accuracy threshold used for the duality-gap "
                    "stopping criterion."),
    pybind11::arg_v("eps_duality_gap_rel",
                    nullopt,
                    "relative accuracy threshold used for the duality-gap "
                    "stopping criterion."),
    pybind11::arg_v("primal_infeasibility_solving",
                    false,
                    "solves the closest feasible problem in L2 sense "
                    "if the QP problem appears to be infeasible."),
    pybind11::arg_v("default_H_eigenvalue_estimate",
                    0.,
                    "Default estimate of the minimal eigen value of H."));
}

} // namespace python
} // namespace sparse
} // namespace proxqp
} // namespace proxsuite
