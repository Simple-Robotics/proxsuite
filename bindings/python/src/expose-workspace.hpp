//
// Copyright (c) 2022-2024 INRIA
//
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <proxsuite/proxqp/dense/workspace.hpp>
#include <proxsuite/proxqp/dense/utils.hpp>

#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <proxsuite/serialization/workspace.hpp>

namespace proxsuite {
namespace proxqp {
namespace dense {
namespace python {
template<typename T>
void
exposeWorkspaceDense(nanobind::module_ m)
{
  ::nanobind::class_<proxsuite::proxqp::dense::Workspace<T>>(m, "workspace")
    .def(::nanobind::init<i64, i64, i64>(),
         nanobind::arg("n") = 0,
         nanobind::arg("n_eq") = 0,
         nanobind::arg("n_in") = 0,
         "Constructor using QP model dimensions.") // constructor)
    .def_ro("H_scaled", &Workspace<T>::H_scaled)
    .def_ro("g_scaled", &Workspace<T>::g_scaled)
    .def_ro("A_scaled", &Workspace<T>::A_scaled)
    .def_ro("C_scaled", &Workspace<T>::C_scaled)
    .def_ro("b_scaled", &Workspace<T>::b_scaled)
    .def_ro("u_scaled", &Workspace<T>::u_scaled)
    .def_ro("l_scaled", &Workspace<T>::l_scaled)
    .def_ro("x_prev", &Workspace<T>::x_prev)
    .def_ro("y_prev", &Workspace<T>::y_prev)
    .def_ro("z_prev", &Workspace<T>::z_prev)
    .def_ro("kkt", &Workspace<T>::kkt)
    .def_ro("current_bijection_map", &Workspace<T>::current_bijection_map)
    .def_ro("new_bijection_map", &Workspace<T>::new_bijection_map)
    .def_ro("active_set_up", &Workspace<T>::active_set_up)
    .def_ro("active_set_low", &Workspace<T>::active_set_low)
    .def_ro("active_inequalities", &Workspace<T>::active_inequalities)
    .def_ro("Hdx", &Workspace<T>::Hdx)
    .def_ro("Cdx", &Workspace<T>::Cdx)
    .def_ro("Adx", &Workspace<T>::Adx)
    .def_ro("active_part_z", &Workspace<T>::active_part_z)
    .def_ro("alphas", &Workspace<T>::alphas)
    .def_ro("dw_aug", &Workspace<T>::dw_aug)
    .def_ro("rhs", &Workspace<T>::rhs)
    .def_ro("err", &Workspace<T>::err)
    .def_ro("dual_feasibility_rhs_2", &Workspace<T>::dual_feasibility_rhs_2)
    .def_ro("correction_guess_rhs_g", &Workspace<T>::correction_guess_rhs_g)
    .def_ro("correction_guess_rhs_b", &Workspace<T>::correction_guess_rhs_b)
    .def_ro("alpha", &Workspace<T>::alpha)
    .def_ro("dual_residual_scaled", &Workspace<T>::dual_residual_scaled)
    .def_ro("primal_residual_in_scaled_up",
            &Workspace<T>::primal_residual_in_scaled_up)
    .def_ro("primal_residual_in_scaled_up_plus_alphaCdx",
            &Workspace<T>::primal_residual_in_scaled_up_plus_alphaCdx)
    .def_ro("primal_residual_in_scaled_low_plus_alphaCdx",
            &Workspace<T>::primal_residual_in_scaled_low_plus_alphaCdx)
    .def_ro("CTz", &Workspace<T>::CTz)
    .def_ro("constraints_changed", &Workspace<T>::constraints_changed)
    .def_ro("dirty", &Workspace<T>::dirty)
    .def_ro("refactorize", &Workspace<T>::refactorize)
    .def_ro("proximal_parameter_update",
            &Workspace<T>::proximal_parameter_update)
    .def_ro("is_initialized", &Workspace<T>::is_initialized)
    .def_ro("n_c", &Workspace<T>::n_c)
    .def("__getstate__",
         [](const Workspace<T>& workspace) {
           return proxsuite::serialization::saveToString(workspace);
         })
    .def("__setstate__", [](Workspace<T>& workspace, nanobind::bytes& s) {
      new (&workspace) Workspace<T>{};
      proxsuite::serialization::loadFromString(workspace, s.c_str());
    });

  ;
}
} // namespace python
} // namespace dense
} // namespace proxqp
} // namespace proxsuite
