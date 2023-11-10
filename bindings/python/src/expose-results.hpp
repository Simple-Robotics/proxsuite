//
// Copyright (c) 2022-2023 INRIA
//
#include <proxsuite/proxqp/results.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/results.hpp>

#include "helpers.hpp"
#include "optional.hpp"

namespace proxsuite {
namespace proxqp {
namespace python {

template<typename T>
void
exposeResults(pybind11::module_ m)
{
  ::pybind11::enum_<QPSolverOutput>(
    m, "QPSolverOutput", pybind11::module_local())
    .value("PROXQP_SOLVED", QPSolverOutput::PROXQP_SOLVED)
    .value("PROXQP_MAX_ITER_REACHED", QPSolverOutput::PROXQP_MAX_ITER_REACHED)
    .value("PROXQP_PRIMAL_INFEASIBLE", QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE)
    .value("PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE",
           QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE)
    .value("PROXQP_DUAL_INFEASIBLE", QPSolverOutput::PROXQP_DUAL_INFEASIBLE)
    .value("PROXQP_NOT_RUN", QPSolverOutput::PROXQP_NOT_RUN)
    .export_values();

  ::pybind11::class_<Info<T>>(m, "Info", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.")
    .def_readwrite("mu_eq", &Info<T>::mu_eq)
    .def_readwrite("mu_in", &Info<T>::mu_in)
    .def_readwrite("rho", &Info<T>::rho)
    .def_readwrite("iter", &Info<T>::iter)
    .def_readwrite("iter_ext", &Info<T>::iter_ext)
    .def_readwrite("run_time", &Info<T>::run_time)
    .def_readwrite("setup_time", &Info<T>::setup_time)
    .def_readwrite("solve_time", &Info<T>::solve_time)
    .def_readwrite("duality_gap", &Info<T>::duality_gap)
    .def_readwrite("pri_res", &Info<T>::pri_res)
    .def_readwrite("dua_res", &Info<T>::dua_res)
    .def_readwrite("duality_gap", &Info<T>::duality_gap)
    .def_readwrite("iterative_residual", &Info<T>::iterative_residual)
    .def_readwrite("objValue", &Info<T>::objValue)
    .def_readwrite("status", &Info<T>::status)
    .def_readwrite("rho_updates", &Info<T>::rho_updates)
    .def_readwrite("mu_updates", &Info<T>::mu_updates)
    .def_readwrite("sparse_backend",
                   &Info<T>::sparse_backend,
                   "Sparse backend used to solve the qp, either SparseCholesky "
                   "or MatrixFree.")
    .def_readwrite("minimal_H_eigenvalue_estimate",
                   &Info<T>::minimal_H_eigenvalue_estimate,
                   "By default it equals 0, in order to get an estimate, set "
                   "appropriately the setting option "
                   "find_H_minimal_eigenvalue.");

  ::pybind11::class_<Results<T>>(m, "Results", pybind11::module_local())
    .def(::pybind11::init<i64, i64, i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         "Constructor from QP model dimensions.") // constructor
    .PROXSUITE_PYTHON_EIGEN_READWRITE(Results<T>, x, "The primal solution.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      Results<T>,
      y,
      "The dual solution associated to the equality constraints.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      Results<T>,
      z,
      "The dual solution associated to the inequality constraints.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      Results<T>,
      se,
      "Optimal shift to the closest feasible problem wrt equality constraints.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(Results<T>,
                                      si,
                                      "Pptimal shift to the closest feasible "
                                      "problem wrt inequality constraints.")
    .def_readwrite("info", &Results<T>::info)
    .def(pybind11::self == pybind11::self)
    .def(pybind11::self != pybind11::self)
    .def(pybind11::pickle(

      [](const Results<T>& results) {
        return pybind11::bytes(proxsuite::serialization::saveToString(results));
      },
      [](pybind11::bytes& s) {
        Results<T> results;
        proxsuite::serialization::loadFromString(results, s);
        return results;
      }));
  ;
}
} // namespace python
} // namespace proxqp
} // namespace proxsuite
