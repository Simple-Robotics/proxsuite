//
// Copyright (c) 2022-2024 INRIA
//
#include <proxsuite/proxqp/results.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/operators.h>
#include "optional-eigen-fix.hpp"

#include <proxsuite/helpers/optional.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/results.hpp>

namespace proxsuite {
namespace proxqp {
namespace python {

template<typename T>
void
exposeResults(nanobind::module_ m)
{
  ::nanobind::enum_<QPSolverOutput>(m, "QPSolverOutput")
    .value("PROXQP_SOLVED", QPSolverOutput::PROXQP_SOLVED)
    .value("PROXQP_MAX_ITER_REACHED", QPSolverOutput::PROXQP_MAX_ITER_REACHED)
    .value("PROXQP_PRIMAL_INFEASIBLE", QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE)
    .value("PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE",
           QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE)
    .value("PROXQP_DUAL_INFEASIBLE", QPSolverOutput::PROXQP_DUAL_INFEASIBLE)
    .value("PROXQP_NOT_RUN", QPSolverOutput::PROXQP_NOT_RUN)
    .export_values();

  ::nanobind::class_<Info<T>>(m, "Info")
    .def(::nanobind::init(), "Default constructor.")
    .def_rw("mu_eq", &Info<T>::mu_eq)
    .def_rw("mu_in", &Info<T>::mu_in)
    .def_rw("rho", &Info<T>::rho)
    .def_rw("iter", &Info<T>::iter)
    .def_rw("iter_ext", &Info<T>::iter_ext)
    .def_rw("run_time", &Info<T>::run_time)
    .def_rw("setup_time", &Info<T>::setup_time)
    .def_rw("solve_time", &Info<T>::solve_time)
    .def_rw("duality_gap", &Info<T>::duality_gap)
    .def_rw("pri_res", &Info<T>::pri_res)
    .def_rw("dua_res", &Info<T>::dua_res)
    .def_rw("duality_gap", &Info<T>::duality_gap)
    .def_rw("iterative_residual", &Info<T>::iterative_residual)
    .def_rw("objValue", &Info<T>::objValue)
    .def_rw("status", &Info<T>::status)
    .def_rw("rho_updates", &Info<T>::rho_updates)
    .def_rw("mu_updates", &Info<T>::mu_updates)
    .def_rw("sparse_backend",
            &Info<T>::sparse_backend,
            "Sparse backend used to solve the qp, either SparseCholesky "
            "or MatrixFree.")
    .def_rw("minimal_H_eigenvalue_estimate",
            &Info<T>::minimal_H_eigenvalue_estimate,
            "By default it equals 0, in order to get an estimate, set "
            "appropriately the setting option "
            "find_H_minimal_eigenvalue.");

  ::nanobind::class_<Results<T>>(m, "Results")
    .def(::nanobind::init<isize, isize, isize>(),
         nanobind::arg("n") = 0,
         nanobind::arg("n_eq") = 0,
         nanobind::arg("n_in") = 0,
         "Constructor from QP model dimensions.") // constructor
    // .PROXSUITE_PYTHON_EIGEN_READWRITE(Results<T>, x, "The primal solution.")
    // .PROXSUITE_PYTHON_EIGEN_READWRITE(
    //   Results<T>,
    //   y,
    //   "The dual solution associated to the equality constraints.")
    // .PROXSUITE_PYTHON_EIGEN_READWRITE(
    //   Results<T>,
    //   z,
    //   "The dual solution associated to the inequality constraints.")
    // .PROXSUITE_PYTHON_EIGEN_READWRITE(
    //   Results<T>,
    //   se,
    //   "Optimal shift to the closest feasible problem wrt equality
    //   constraints.")
    // .PROXSUITE_PYTHON_EIGEN_READWRITE(Results<T>,
    //                                   si,
    //                                   "Optimal shift to the closest feasible
    //                                   " "problem wrt inequality
    //                                   constraints.")
    .def_rw("x", &Results<T>::x, "The primal solution.")
    .def_rw("y",
            &Results<T>::y,
            "The dual solution associated to the equality constraints.")
    .def_rw("z",
            &Results<T>::z,
            "The dual solution associated to the inequality constraints.")
    .def_rw(
      "se",
      &Results<T>::se,
      "Optimal shift to the closest feasible problem wrt equality constraints.")
    .def_rw("si",
            &Results<T>::si,
            "Optimal shift to the closest feasible problem wrt inequality "
            "constraints.")
    .def_rw("info", &Results<T>::info)
    .def(nanobind::self == nanobind::self)
    .def(nanobind::self != nanobind::self)
    .def("__getstate__",
         [](const Results<T>& results) {
           return proxsuite::serialization::saveToString(results);
         })
    .def("__setstate__", [](Results<T>& results, const std::string& s) {
      new (&results) Results<T>{};
      proxsuite::serialization::loadFromString(results, s);
    });
  ;
}
} // namespace python
} // namespace proxqp
} // namespace proxsuite
