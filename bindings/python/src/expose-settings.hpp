//
// Copyright (c) 2022-2024 INRIA
//
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/operators.h>

#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/settings.hpp>

namespace proxsuite {
namespace proxqp {
namespace python {
template<typename T>
void
exposeSettings(nanobind::module_ m)
{

  ::nanobind::enum_<InitialGuessStatus>(m, "InitialGuess")
    .value("NO_INITIAL_GUESS", InitialGuessStatus::NO_INITIAL_GUESS)
    .value("EQUALITY_CONSTRAINED_INITIAL_GUESS",
           InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS)
    .value("WARM_START_WITH_PREVIOUS_RESULT",
           InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT)
    .value("WARM_START", InitialGuessStatus::WARM_START)
    .value("COLD_START_WITH_PREVIOUS_RESULT",
           InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT)
    .export_values();

  ::nanobind::enum_<MeritFunctionType>(m, "MeritFunctionType")
    .value("GPDAL", MeritFunctionType::GPDAL)
    .value("PDAL", MeritFunctionType::PDAL)
    .export_values();

  ::nanobind::enum_<SparseBackend>(m, "SparseBackend")
    .value("Automatic", SparseBackend::Automatic)
    .value("MatrixFree", SparseBackend::MatrixFree)
    .value("SparseCholesky", SparseBackend::SparseCholesky)
    .export_values();
  ::nanobind::enum_<EigenValueEstimateMethodOption>(
    m, "EigenValueEstimateMethodOption")
    .value("PowerIteration", EigenValueEstimateMethodOption::PowerIteration)
    .value("ExactMethod", EigenValueEstimateMethodOption::ExactMethod)
    .export_values();

  ::nanobind::class_<Settings<T>>(m, "Settings")
    .def(::nanobind::init(), "Default constructor.") // constructor
    .def_rw("default_rho", &Settings<T>::default_rho)
    .def_rw("default_mu_eq", &Settings<T>::default_mu_eq)
    .def_rw("default_mu_in", &Settings<T>::default_mu_in)
    .def_rw("alpha_bcl", &Settings<T>::alpha_bcl)
    .def_rw("beta_bcl", &Settings<T>::beta_bcl)
    .def_rw("refactor_dual_feasibility_threshold",
            &Settings<T>::refactor_dual_feasibility_threshold)
    .def_rw("refactor_rho_threshold", &Settings<T>::refactor_rho_threshold)
    .def_rw("mu_min_eq", &Settings<T>::mu_min_eq)
    .def_rw("mu_min_in", &Settings<T>::mu_min_in)
    .def_rw("mu_max_eq_inv", &Settings<T>::mu_max_eq_inv)
    .def_rw("mu_max_in_inv", &Settings<T>::mu_max_in_inv)
    .def_rw("mu_update_factor", &Settings<T>::mu_update_factor)
    .def_rw("cold_reset_mu_eq", &Settings<T>::cold_reset_mu_eq)
    .def_rw("cold_reset_mu_in", &Settings<T>::cold_reset_mu_in)
    .def_rw("max_iter", &Settings<T>::max_iter)
    .def_rw("max_iter_in", &Settings<T>::max_iter_in)
    .def_rw("eps_abs", &Settings<T>::eps_abs)
    .def_rw("eps_rel", &Settings<T>::eps_rel)
    .def_rw("eps_primal_inf", &Settings<T>::eps_primal_inf)
    .def_rw("eps_dual_inf", &Settings<T>::eps_dual_inf)
    .def_rw("nb_iterative_refinement", &Settings<T>::nb_iterative_refinement)
    .def_rw("initial_guess", &Settings<T>::initial_guess)
    .def_rw("sparse_backend", &Settings<T>::sparse_backend)
    .def_rw("preconditioner_accuracy", &Settings<T>::preconditioner_accuracy)
    .def_rw("preconditioner_max_iter", &Settings<T>::preconditioner_max_iter)
    .def_rw("compute_timings", &Settings<T>::compute_timings)
    .def_rw("compute_preconditioner", &Settings<T>::compute_preconditioner)
    .def_rw("update_preconditioner", &Settings<T>::update_preconditioner)
    .def_rw("check_duality_gap", &Settings<T>::check_duality_gap)
    .def_rw("eps_duality_gap_abs", &Settings<T>::eps_duality_gap_abs)
    .def_rw("eps_duality_gap_rel", &Settings<T>::eps_duality_gap_rel)
    .def_rw("verbose", &Settings<T>::verbose)
    .def_rw("bcl_update", &Settings<T>::bcl_update)
    .def_rw("merit_function_type", &Settings<T>::merit_function_type)
    .def_rw("alpha_gpdal", &Settings<T>::alpha_gpdal)
    .def_rw("primal_infeasibility_solving",
            &Settings<T>::primal_infeasibility_solving)
    .def_rw("frequence_infeasibility_check",
            &Settings<T>::frequence_infeasibility_check)
    .def_rw("default_H_eigenvalue_estimate",
            &Settings<T>::default_H_eigenvalue_estimate)
    .def(nanobind::self == nanobind::self)
    .def(nanobind::self != nanobind::self)
    .def("__getstate__",
         [](const Settings<T>& settings) {
           return proxsuite::serialization::saveToString(settings);
         })
    .def("__setstate__", [](Settings<T>& settings, const std::string& s) {
      new (&settings) Settings<T>{};
      proxsuite::serialization::loadFromString(settings, s);
    });
  ;
}
} // namespace python
} // namespace proxqp
} // namespace proxsuite
