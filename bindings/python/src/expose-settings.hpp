//
// Copyright (c) 2022 INRIA
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/settings.hpp>

namespace proxsuite {
namespace proxqp {
namespace python {
template<typename T>
void
exposeSettings(pybind11::module_ m)
{

  ::pybind11::enum_<InitialGuessStatus>(
    m, "InitialGuess", pybind11::module_local())
    .value("NO_INITIAL_GUESS", InitialGuessStatus::NO_INITIAL_GUESS)
    .value("EQUALITY_CONSTRAINED_INITIAL_GUESS",
           InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS)
    .value("WARM_START_WITH_PREVIOUS_RESULT",
           InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT)
    .value("WARM_START", InitialGuessStatus::WARM_START)
    .value("COLD_START_WITH_PREVIOUS_RESULT",
           InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT)
    .export_values();

  ::pybind11::enum_<SparseBackend>(m, "SparseBackend", pybind11::module_local())
    .value("Automatic", SparseBackend::Automatic)
    .value("MatrixFree", SparseBackend::MatrixFree)
    .value("SparseCholesky", SparseBackend::SparseCholesky)
    .export_values();

  ::pybind11::class_<Settings<T>>(m, "Settings", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.") // constructor
    .def_readwrite("default_rho", &Settings<T>::default_rho)
    .def_readwrite("default_mu_eq", &Settings<T>::default_mu_eq)
    .def_readwrite("default_mu_in", &Settings<T>::default_mu_in)
    .def_readwrite("alpha_bcl", &Settings<T>::alpha_bcl)
    .def_readwrite("beta_bcl", &Settings<T>::beta_bcl)
    .def_readwrite("refactor_dual_feasibility_threshold",
                   &Settings<T>::refactor_dual_feasibility_threshold)
    .def_readwrite("refactor_rho_threshold",
                   &Settings<T>::refactor_rho_threshold)
    .def_readwrite("mu_min_eq", &Settings<T>::mu_min_eq)
    .def_readwrite("mu_min_in", &Settings<T>::mu_min_in)
    .def_readwrite("mu_update_factor", &Settings<T>::mu_update_factor)
    .def_readwrite("cold_reset_mu_eq", &Settings<T>::cold_reset_mu_eq)
    .def_readwrite("cold_reset_mu_in", &Settings<T>::cold_reset_mu_in)
    .def_readwrite("max_iter", &Settings<T>::max_iter)
    .def_readwrite("max_iter_in", &Settings<T>::max_iter_in)
    .def_readwrite("eps_abs", &Settings<T>::eps_abs)
    .def_readwrite("eps_rel", &Settings<T>::eps_rel)
    .def_readwrite("eps_primal_inf", &Settings<T>::eps_primal_inf)
    .def_readwrite("eps_dual_inf", &Settings<T>::eps_dual_inf)
    .def_readwrite("nb_iterative_refinement",
                   &Settings<T>::nb_iterative_refinement)
    .def_readwrite("initial_guess", &Settings<T>::initial_guess)
    .def_readwrite("sparse_backend", &Settings<T>::sparse_backend)
    .def_readwrite("preconditioner_accuracy",
                   &Settings<T>::preconditioner_accuracy)
    .def_readwrite("preconditioner_max_iter",
                   &Settings<T>::preconditioner_max_iter)
    .def_readwrite("compute_timings", &Settings<T>::compute_timings)
    .def_readwrite("compute_preconditioner",
                   &Settings<T>::compute_preconditioner)
    .def_readwrite("update_preconditioner", &Settings<T>::update_preconditioner)
    .def_readwrite("check_duality_gap", &Settings<T>::check_duality_gap)
    .def_readwrite("eps_duality_gap_abs", &Settings<T>::eps_duality_gap_abs)
    .def_readwrite("eps_duality_gap_rel", &Settings<T>::eps_duality_gap_rel)
    .def_readwrite("verbose", &Settings<T>::verbose)
    .def_readwrite("bcl_update", &Settings<T>::bcl_update)
    .def(pybind11::self == pybind11::self)
    .def(pybind11::self != pybind11::self)
    .def(pybind11::pickle(

      [](const Settings<T>& settings) {
        return pybind11::bytes(
          proxsuite::serialization::saveToString(settings));
      },
      [](pybind11::bytes& s) {
        Settings<T> settings;
        proxsuite::serialization::loadFromString(settings, s);
        return settings;
      }));
  ;
}
} // namespace python
} // namespace proxqp
} // namespace proxsuite
