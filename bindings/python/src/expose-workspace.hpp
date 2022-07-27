//
// Copyright (c) 2022, INRIA
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <proxsuite/proxqp/dense/workspace.hpp>
#include <proxsuite/proxqp/dense/utils.hpp>

namespace proxsuite {
namespace proxqp {
namespace dense {
namespace python {
template <typename T>
void exposeWorkspaceDense(pybind11::module_ m) {
	::pybind11::class_<Workspace<T>>(m, "Workspace")
			.def(::pybind11::init<i64, i64, i64>(),
				pybind11::arg_v("n",0,"primal dimension."),pybind11::arg_v("n_eq",0,"number of equality constraints."),pybind11::arg_v("n_in",0,"number of inequality constraints."),
				"Constructor using QP model dimensions.") // constructor)
			.def_readwrite("H_scaled", &Workspace<T>::H_scaled)
			.def_readwrite("g_scaled", &Workspace<T>::g_scaled)
			.def_readwrite("A_scaled", &Workspace<T>::A_scaled)
			.def_readwrite("C_scaled", &Workspace<T>::C_scaled)
			.def_readwrite("b_scaled", &Workspace<T>::b_scaled)
			.def_readwrite("u_scaled", &Workspace<T>::u_scaled)
			.def_readwrite("l_scaled", &Workspace<T>::l_scaled)
			.def_readwrite("x_prev", &Workspace<T>::x_prev)
			.def_readwrite("y_prev", &Workspace<T>::y_prev)
			.def_readwrite("z_prev", &Workspace<T>::z_prev)
			.def_readwrite("kkt", &Workspace<T>::kkt)
			.def_readwrite(
					"current_bijection_map",
					&Workspace<T>::current_bijection_map)
			.def_readwrite(
					"new_bijection_map", &Workspace<T>::new_bijection_map)
			.def_readwrite("active_set_up", &Workspace<T>::active_set_up)
			.def_readwrite("active_set_low", &Workspace<T>::active_set_low)
			.def_readwrite(
					"active_inequalities", &Workspace<T>::active_inequalities)
			.def_readwrite("Hdx", &Workspace<T>::Hdx)
			.def_readwrite("Cdx", &Workspace<T>::Cdx)
			.def_readwrite("Adx", &Workspace<T>::Adx)
			.def_readwrite("active_part_z", &Workspace<T>::active_part_z)
			.def_readwrite("alphas", &Workspace<T>::alphas)
			.def_readwrite("dw_aug", &Workspace<T>::dw_aug)
			.def_readwrite("rhs", &Workspace<T>::rhs)
			.def_readwrite("err", &Workspace<T>::err)
			.def_readwrite(
					"primal_feasibility_rhs_1_eq",
					&Workspace<T>::primal_feasibility_rhs_1_eq)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_u",
					&Workspace<T>::primal_feasibility_rhs_1_in_u)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_l",
					&Workspace<T>::primal_feasibility_rhs_1_in_l)
			.def_readwrite(
					"dual_feasibility_rhs_2",
					&Workspace<T>::dual_feasibility_rhs_2)
			.def_readwrite(
					"correction_guess_rhs_g",
					&Workspace<T>::correction_guess_rhs_g)
			.def_readwrite("alpha", &Workspace<T>::alpha)
			.def_readwrite(
					"dual_residual_scaled",
					&Workspace<T>::dual_residual_scaled)
			.def_readwrite(
					"primal_residual_eq_scaled",
					&Workspace<T>::primal_residual_eq_scaled)
			.def_readwrite(
					"primal_residual_in_scaled_up",
					&Workspace<T>::primal_residual_in_scaled_up)
			.def_readwrite(
					"primal_residual_in_scaled_low",
					&Workspace<T>::primal_residual_in_scaled_low)
			.def_readwrite(
					"primal_residual_in_scaled_up_plus_alphaCdx",
					&Workspace<T>::primal_residual_in_scaled_up_plus_alphaCdx)
			.def_readwrite(
					"primal_residual_in_scaled_low_plus_alphaCdx",
					&Workspace<T>::primal_residual_in_scaled_low_plus_alphaCdx)
			.def_readwrite("CTz", &Workspace<T>::CTz);
}
} // namespace python
} // namespace dense
} // namespace proxqp
} // namespace proxsuite
