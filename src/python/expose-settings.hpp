#include <qp/Settings.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace proxsuite {
namespace qp {
namespace python {
template <typename T>
void exposeSettings(pybind11::module_ m) {
	::pybind11::class_<qp::Settings<T>>(m, "Settings")
			.def(::pybind11::init()) // constructor
			.def_readwrite("alpha_bcl", &qp::Settings<T>::alpha_bcl)
			.def_readwrite("beta_bcl", &qp::Settings<T>::beta_bcl)
			.def_readwrite(
					"refactor_dual_feasibility_threshold",
					&qp::Settings<T>::refactor_dual_feasibility_threshold)
			.def_readwrite(
					"refactor_rho_threshold", &qp::Settings<T>::refactor_rho_threshold)
			.def_readwrite("mu_max_eq", &qp::Settings<T>::mu_max_eq)
			.def_readwrite("mu_max_in", &qp::Settings<T>::mu_max_in)
			.def_readwrite("mu_update_factor", &qp::Settings<T>::mu_update_factor)
			.def_readwrite("cold_reset_mu_eq", &qp::Settings<T>::cold_reset_mu_eq)
			.def_readwrite("cold_reset_mu_in", &qp::Settings<T>::cold_reset_mu_in)
			.def_readwrite("max_iter", &qp::Settings<T>::max_iter)
			.def_readwrite("max_iter_in", &qp::Settings<T>::max_iter_in)
			.def_readwrite("eps_abs", &qp::Settings<T>::eps_abs)
			.def_readwrite("eps_rel", &qp::Settings<T>::eps_rel)
			.def_readwrite("eps_primal_inf", &qp::Settings<T>::eps_primal_inf)
			.def_readwrite("eps_dual_inf", &qp::Settings<T>::eps_dual_inf)
			.def_readwrite(
					"nb_iterative_refinement", &qp::Settings<T>::nb_iterative_refinement)
			.def_readwrite("warm_start", &qp::Settings<T>::warm_start)
			.def_readwrite("verbose", &qp::Settings<T>::verbose);
}
} // namespace python
} // namespace qp
} // namespace proxsuite
