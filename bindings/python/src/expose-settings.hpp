//
// Copyright (c) 2022, INRIA
//
#include <proxsuite/qp/settings.hpp>
#include <proxsuite/qp/status.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace proxsuite {
namespace qp {
namespace python {
template <typename T>
void exposeSettings(pybind11::module_ m) {

	::pybind11::enum_<InitialGuessStatus>(m, "InitialGuess",pybind11::module_local())
		.value("NO_INITIAL_GUESS", InitialGuessStatus::NO_INITIAL_GUESS)
		.value("EQUALITY_CONSTRAINED_INITIAL_GUESS", InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS)
		.value("WARM_START_WITH_PREVIOUS_RESULT", InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT)
		.value("WARM_START", InitialGuessStatus::WARM_START)
		.value("COLD_START_WITH_PREVIOUS_RESULT", InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT)
		.export_values();

	::pybind11::class_<Settings<T>>(m, "Settings",pybind11::module_local())
			.def(::pybind11::init(),"Default constructor.") // constructor
			.def_readwrite("alpha_bcl", &Settings<T>::alpha_bcl)
			.def_readwrite("beta_bcl", &Settings<T>::beta_bcl)
			.def_readwrite(
					"refactor_dual_feasibility_threshold",
					&Settings<T>::refactor_dual_feasibility_threshold)
			.def_readwrite(
					"refactor_rho_threshold", &Settings<T>::refactor_rho_threshold)
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
			.def_readwrite("nb_iterative_refinement", &Settings<T>::nb_iterative_refinement)
			.def_readwrite("initial_guess", &Settings<T>::initial_guess)
			.def_readwrite("preconditioner_accuracy", &Settings<T>::preconditioner_accuracy)
			.def_readwrite("preconditioner_max_iter", &Settings<T>::preconditioner_max_iter)
			.def_readwrite("compute_timings", &Settings<T>::compute_timings)
			.def_readwrite("compute_preconditioner", &Settings<T>::compute_preconditioner)
			.def_readwrite("update_preconditioner", &Settings<T>::update_preconditioner)
			.def_readwrite("verbose", &Settings<T>::verbose)
			.def_readwrite("bcl_update", &Settings<T>::bcl_update)
			;
}
} // namespace python
} // namespace qp
} // namespace proxsuite
