//
// Copyright (c) 2022 INRIA
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <proxsuite/proxqp/dense/workspace.hpp>
#include <proxsuite/proxqp/dense/utils.hpp>

#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <proxsuite/serialization/workspace.hpp>
#include "helpers.hpp"
namespace proxsuite {
namespace proxqp {
namespace dense {
namespace python {
template<typename T>
void
exposeWorkspaceDense(pybind11::module_ m)
{
  ::pybind11::class_<proxsuite::proxqp::dense::Workspace<T>>(
    m, "workspace", pybind11::module_local())
    .def(::pybind11::init<i64, i64, i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         "Constructor using QP model dimensions.") // constructor)
    .def_readonly("H_scaled", &Workspace<T>::H_scaled)
    .def_readonly("g_scaled", &Workspace<T>::g_scaled)
    .def_readonly("A_scaled", &Workspace<T>::A_scaled)
    .def_readonly("C_scaled", &Workspace<T>::C_scaled)
    .def_readonly("b_scaled", &Workspace<T>::b_scaled)
    .def_readonly("u_scaled", &Workspace<T>::u_scaled)
    .def_readonly("l_scaled", &Workspace<T>::l_scaled)
    .def_readonly("x_prev", &Workspace<T>::x_prev)
    .def_readonly("y_prev", &Workspace<T>::y_prev)
    .def_readonly("z_prev", &Workspace<T>::z_prev)
    .def_readonly("kkt", &Workspace<T>::kkt)
    .def_readonly("current_bijection_map", &Workspace<T>::current_bijection_map)
    .def_readonly("new_bijection_map", &Workspace<T>::new_bijection_map)
    .def_readonly("active_set_up", &Workspace<T>::active_set_up)
    .def_readonly("active_set_low", &Workspace<T>::active_set_low)
    .def_readonly("active_inequalities", &Workspace<T>::active_inequalities)
    .def_readonly("Hdx", &Workspace<T>::Hdx)
    .def_readonly("Cdx", &Workspace<T>::Cdx)
    .def_readonly("Adx", &Workspace<T>::Adx)
    .def_readonly("active_part_z", &Workspace<T>::active_part_z)
    .def_readonly("alphas", &Workspace<T>::alphas)
    .def_readonly("dw_aug", &Workspace<T>::dw_aug)
    .def_readonly("rhs", &Workspace<T>::rhs)
    .def_readonly("err", &Workspace<T>::err)
    .def_readonly("dual_feasibility_rhs_2",
                  &Workspace<T>::dual_feasibility_rhs_2)
    .def_readonly("correction_guess_rhs_g",
                  &Workspace<T>::correction_guess_rhs_g)
    .def_readonly("correction_guess_rhs_b",
                  &Workspace<T>::correction_guess_rhs_b)
    .def_readonly("alpha", &Workspace<T>::alpha)
    .def_readonly("dual_residual_scaled", &Workspace<T>::dual_residual_scaled)
    .def_readonly("primal_residual_in_scaled_up",
                  &Workspace<T>::primal_residual_in_scaled_up)
    .def_readonly("primal_residual_in_scaled_up_plus_alphaCdx",
                  &Workspace<T>::primal_residual_in_scaled_up_plus_alphaCdx)
    .def_readonly("primal_residual_in_scaled_low_plus_alphaCdx",
                  &Workspace<T>::primal_residual_in_scaled_low_plus_alphaCdx)
    .def_readonly("CTz", &Workspace<T>::CTz)
    .def_readonly("constraints_changed", &Workspace<T>::constraints_changed)
    .def_readonly("dirty", &Workspace<T>::dirty)
    .def_readonly("refactorize", &Workspace<T>::refactorize)
    .def_readonly("proximal_parameter_update",
                  &Workspace<T>::proximal_parameter_update)
    .def_readonly("is_initialized", &Workspace<T>::is_initialized)
    .def_readonly("n_c", &Workspace<T>::n_c)
    .def(pybind11::pickle(

      [](const Workspace<T>& workspace) {
        return pybind11::bytes(
          proxsuite::serialization::saveToString(workspace));
      },
      [](pybind11::bytes& s) {
        Workspace<T> workspace;
        proxsuite::serialization::loadFromString(workspace, s);
        return workspace;
      }));

  ;
}
} // namespace python
} // namespace dense
} // namespace proxqp
} // namespace proxsuite
