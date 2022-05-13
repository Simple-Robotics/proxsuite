#include <qp/dense/Workspace.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace proxsuite {
namespace qp {
namespace dense {
namespace python {
template <typename T>
void exposeWorkspaceDense(pybind11::module_ m) {
	::pybind11::class_<qp::dense::Workspace<T>>(m, "Workspace")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readwrite("H_scaled", &qp::dense::Workspace<T>::H_scaled)
			.def_readwrite("g_scaled", &qp::dense::Workspace<T>::g_scaled)
			.def_readwrite("A_scaled", &qp::dense::Workspace<T>::A_scaled)
			.def_readwrite("C_scaled", &qp::dense::Workspace<T>::C_scaled)
			.def_readwrite("b_scaled", &qp::dense::Workspace<T>::b_scaled)
			.def_readwrite("u_scaled", &qp::dense::Workspace<T>::u_scaled)
			.def_readwrite("l_scaled", &qp::dense::Workspace<T>::l_scaled)
			.def_readwrite("x_prev", &qp::dense::Workspace<T>::x_prev)
			.def_readwrite("y_prev", &qp::dense::Workspace<T>::y_prev)
			.def_readwrite("z_prev", &qp::dense::Workspace<T>::z_prev)
			.def_readwrite("kkt", &qp::dense::Workspace<T>::kkt)
			.def_readwrite(
					"current_bijection_map",
					&qp::dense::Workspace<T>::current_bijection_map)
			.def_readwrite(
					"new_bijection_map", &qp::dense::Workspace<T>::new_bijection_map)
			.def_readwrite("active_set_up", &qp::dense::Workspace<T>::active_set_up)
			.def_readwrite("active_set_low", &qp::dense::Workspace<T>::active_set_low)
			.def_readwrite(
					"active_inequalities", &qp::dense::Workspace<T>::active_inequalities)
			.def_readwrite("Hdx", &qp::dense::Workspace<T>::Hdx)
			.def_readwrite("Cdx", &qp::dense::Workspace<T>::Cdx)
			.def_readwrite("Adx", &qp::dense::Workspace<T>::Adx)
			.def_readwrite("active_part_z", &qp::dense::Workspace<T>::active_part_z)
			.def_readwrite("alphas", &qp::dense::Workspace<T>::alphas)
			.def_readwrite("dw_aug", &qp::dense::Workspace<T>::dw_aug)
			.def_readwrite("rhs", &qp::dense::Workspace<T>::rhs)
			.def_readwrite("err", &qp::dense::Workspace<T>::err)
			.def_readwrite(
					"primal_feasibility_rhs_1_eq",
					&qp::dense::Workspace<T>::primal_feasibility_rhs_1_eq)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_u",
					&qp::dense::Workspace<T>::primal_feasibility_rhs_1_in_u)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_l",
					&qp::dense::Workspace<T>::primal_feasibility_rhs_1_in_l)
			.def_readwrite(
					"dual_feasibility_rhs_2",
					&qp::dense::Workspace<T>::dual_feasibility_rhs_2)
			.def_readwrite(
					"correction_guess_rhs_g",
					&qp::dense::Workspace<T>::correction_guess_rhs_g)
			.def_readwrite("alpha", &qp::dense::Workspace<T>::alpha)
			.def_readwrite(
					"dual_residual_scaled",
					&qp::dense::Workspace<T>::dual_residual_scaled)
			.def_readwrite(
					"primal_residual_eq_scaled",
					&qp::dense::Workspace<T>::primal_residual_eq_scaled)
			.def_readwrite(
					"primal_residual_in_scaled_up",
					&qp::dense::Workspace<T>::primal_residual_in_scaled_up)
			.def_readwrite(
					"primal_residual_in_scaled_low",
					&qp::dense::Workspace<T>::primal_residual_in_scaled_low)
			.def_readwrite(
					"primal_residual_in_scaled_up_plus_alphaCdx",
					&qp::dense::Workspace<T>::primal_residual_in_scaled_up_plus_alphaCdx)
			.def_readwrite(
					"primal_residual_in_scaled_low_plus_alphaCdx",
					&qp::dense::Workspace<T>::primal_residual_in_scaled_low_plus_alphaCdx)
			.def_readwrite("CTz", &qp::dense::Workspace<T>::CTz);
}
} // namespace python
} // namespace dense
} // namespace qp
} // namespace proxsuite
