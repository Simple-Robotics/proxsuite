//
// Copyright (c) 2022-2024 INRIA
//

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/operators.h>

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/wrapper.hpp>

#include "common/expose.hpp"

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
exposeQpObjectDense(nanobind::module_& m)
{
  ::nanobind::enum_<DenseBackend>(m, "DenseBackend")
    .value("Automatic", DenseBackend::Automatic)
    .value("PrimalDualLDLT", DenseBackend::PrimalDualLDLT)
    .value("PrimalLDLT", DenseBackend::PrimalLDLT)
    .export_values();

  ::nanobind::enum_<HessianType>(m, "HessianType")
    .value("Dense", proxsuite::proxqp::HessianType::Dense)
    .value("Zero", proxsuite::proxqp::HessianType::Zero)
    .value("Diagonal", proxsuite::proxqp::HessianType::Diagonal)
    .export_values();

  proxsuite::common::dense::python::exposeDenseQP<T, dense::QP<T>>(
    m); // (scope: m, name: "QP")
}
} // namespace python
} // namespace dense

namespace sparse {

namespace python {

template<typename T, typename I>
void
exposeQpObjectSparse(nanobind::module_& m)
{

  ::nanobind::class_<sparse::QP<T, I>>(m, "QP")
    .def(::nanobind::init<isize, isize, isize>(),
         nanobind::arg("n") = 0,
         nanobind::arg("n_eq") = 0,
         nanobind::arg("n_in") = 0,
         "Constructor using QP model dimensions.") // constructor
    .def(::nanobind::init<const sparse::SparseMat<bool, I>&,
                          const sparse::SparseMat<bool, I>&,
                          const sparse::SparseMat<bool, I>&>(),
         nanobind::arg("H_mask") = nanobind::none(),
         nanobind::arg("A_mask") = nanobind::none(),
         nanobind::arg("C_mask") = 0,
         "Constructor using QP model sparsity structure.") // constructor
    .def_ro("model", &sparse::QP<T, I>::model, "class containing the QP model")
    .def_rw("results",
            &sparse::QP<T, I>::results,
            "class containing the solution or certificate of infeasibility, "
            "and "
            "information statistics in an info subclass.")
    .def_rw("settings", &sparse::QP<T, I>::settings, "Settings of the solver.")
    .def("init",
         &sparse::QP<T, I>::init,
         "function for initializing the model when passing sparse matrices in "
         "entry.",
         nanobind::arg("H") = nanobind::none(),
         nanobind::arg("g") = nanobind::none(),
         nanobind::arg("A") = nanobind::none(),
         nanobind::arg("b") = nanobind::none(),
         nanobind::arg("C") = nanobind::none(),
         nanobind::arg("l") = nanobind::none(),
         nanobind::arg("u") = nanobind::none(),
         nanobind::arg("compute_preconditioner") = true,
         nanobind::arg("rho") = nanobind::none(),
         nanobind::arg("mu_eq") = nanobind::none(),
         nanobind::arg("mu_in") = nanobind::none(),
         nanobind::arg("manual_minimal_H_eigenvalue") = nanobind::none())

    .def("update",
         &sparse::QP<T, I>::update,
         "function for updating the model when passing sparse matrices in "
         "entry.",
         nanobind::arg("H") = nanobind::none(),
         nanobind::arg("g") = nanobind::none(),
         nanobind::arg("A") = nanobind::none(),
         nanobind::arg("b") = nanobind::none(),
         nanobind::arg("C") = nanobind::none(),
         nanobind::arg("l") = nanobind::none(),
         nanobind::arg("u") = nanobind::none(),
         nanobind::arg("update_preconditioner") = false,
         nanobind::arg("rho") = nanobind::none(),
         nanobind::arg("mu_eq") = nanobind::none(),
         nanobind::arg("mu_in") = nanobind::none(),
         nanobind::arg("manual_minimal_H_eigenvalue") = nanobind::none())
    .def("solve",
         static_cast<void (sparse::QP<T, I>::*)()>(&sparse::QP<T, I>::solve),
         "function used for solving the QP problem, using default parameters.")
    .def("solve",
         static_cast<void (sparse::QP<T, I>::*)(optional<sparse::VecRef<T>> x,
                                                optional<sparse::VecRef<T>> y,
                                                optional<sparse::VecRef<T>> z)>(
           &sparse::QP<T, I>::solve),
         "function used for solving the QP problem, when passing a warm start.")
    .def("cleanup",
         &sparse::QP<T, I>::cleanup,
         "function used for cleaning the result "
         "class.");
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
