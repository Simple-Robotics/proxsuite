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
    pybind11::overload_cast<std::optional<dense::MatRef<T>>,
                            std::optional<dense::VecRef<T>>,
                            std::optional<dense::MatRef<T>>,
                            std::optional<dense::VecRef<T>>,
                            std::optional<dense::MatRef<T>>,
                            std::optional<dense::VecRef<T>>,
                            std::optional<dense::VecRef<T>>,
                            std::optional<VecRef<T>>,
                            std::optional<VecRef<T>>,
                            std::optional<VecRef<T>>,
                            std::optional<T>,
                            std::optional<T>,
                            std::optional<T>,
                            std::optional<T>,
                            std::optional<T>,
                            std::optional<bool>,
                            bool,
                            bool,
                            std::optional<isize>,
                            proxsuite::proxqp::InitialGuessStatus>(
      &dense::solve<T>),
    "Function for solving a QP problem using PROXQP sparse backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    pybind11::arg_v("H", std::nullopt, "quadratic cost with dense format."),
    pybind11::arg_v("g", std::nullopt, "linear cost"),
    pybind11::arg_v(
      "A", std::nullopt, "equality constraint matrix with dense format."),
    pybind11::arg_v("b", std::nullopt, "equality constraint vector"),
    pybind11::arg_v(
      "C", std::nullopt, "inequality constraint matrix with dense format."),
    pybind11::arg_v("l", std::nullopt, "upper inequality constraint vector"),
    pybind11::arg_v("u", std::nullopt, "lower inequality constraint vector"),
    pybind11::arg_v("x", std::nullopt, "primal warm start"),
    pybind11::arg_v("y", std::nullopt, "dual equality warm start"),
    pybind11::arg_v("z", std::nullopt, "dual inequality warm start"),
    pybind11::arg_v(
      "eps_abs",
      std::nullopt,
      "absolute accuracy level used for the solver stopping criterion."),
    pybind11::arg_v("eps_rel",
                    std::nullopt,
                    "relative accuracy level used for the solver stopping "
                    "criterion. Deactivated in standard settings."),
    pybind11::arg_v("rho", std::nullopt, "primal proximal parameter"),
    pybind11::arg_v(
      "mu_eq", std::nullopt, "dual equality constraint proximal parameter"),
    pybind11::arg_v(
      "mu_in", std::nullopt, "dual inequality constraint proximal parameter"),
    pybind11::arg_v("verbose",
                    std::nullopt,
                    "verbose option to print information at each iteration."),
    pybind11::arg_v("compute_preconditioner",
                    true,
                    "executes the default preconditioner for reducing ill "
                    "conditioning and speeding up the solver."),
    pybind11::arg_v("compute_timings", true, "compute solver's timings."),
    pybind11::arg_v("max_iter", std::nullopt, "maximum number of iteration."),
    pybind11::arg_v(
      "initial_guess",
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      "maximum number of iteration."));
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
    "Function for solving a QP problem using PROXQP dense backend directly "
    "without defining a QP object. It is possible to set up some of the solver "
    "parameters (warm start, initial guess option, proximal step sizes, "
    "absolute and relative accuracies, maximum number of iterations, "
    "preconditioner execution).",
    pybind11::arg_v("H", std::nullopt, "quadratic cost with dense format."),
    pybind11::arg_v("g", std::nullopt, "linear cost"),
    pybind11::arg_v(
      "A", std::nullopt, "equality constraint matrix with dense format."),
    pybind11::arg_v("b", std::nullopt, "equality constraint vector"),
    pybind11::arg_v(
      "C", std::nullopt, "inequality constraint matrix with dense format."),
    pybind11::arg_v("l", std::nullopt, "upper inequality constraint vector"),
    pybind11::arg_v("u", std::nullopt, "lower inequality constraint vector"),
    pybind11::arg_v("x", std::nullopt, "primal warm start"),
    pybind11::arg_v("y", std::nullopt, "dual equality warm start"),
    pybind11::arg_v("z", std::nullopt, "dual inequality warm start"),
    pybind11::arg_v(
      "eps_abs",
      std::nullopt,
      "absolute accuracy level used for the solver stopping criterion."),
    pybind11::arg_v("eps_rel",
                    std::nullopt,
                    "relative accuracy level used for the solver stopping "
                    "criterion. Deactivated in standard settings."),
    pybind11::arg_v("rho", std::nullopt, "primal proximal parameter"),
    pybind11::arg_v(
      "mu_eq", std::nullopt, "dual equality constraint proximal parameter"),
    pybind11::arg_v(
      "mu_in", std::nullopt, "dual inequality constraint proximal parameter"),
    pybind11::arg_v("verbose",
                    std::nullopt,
                    "verbose option to print information at each iteration."),
    pybind11::arg_v("compute_preconditioner",
                    true,
                    "executes the default preconditioner for reducing ill "
                    "conditioning and speeding up the solver."),
    pybind11::arg_v("compute_timings", true, "compute solver's timings."),
    pybind11::arg_v("max_iter", std::nullopt, "maximum number of iteration."),
    pybind11::arg_v(
      "initial_guess",
      proxsuite::proxqp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS,
      "maximum number of iteration."));
}

} // namespace python
} // namespace sparse
} // namespace proxqp
} // namespace proxsuite
