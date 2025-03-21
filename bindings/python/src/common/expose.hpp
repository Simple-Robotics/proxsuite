//
// Copyright (c) 2022-2024 INRIA
//

#ifndef bindings_python_src_common_expose
#define bindings_python_src_common_expose

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/operators.h>

#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/wrapper.hpp>

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/osqp/dense/wrapper.hpp>

#include <proxsuite/solvers/common/utils.hpp>

namespace proxsuite {
namespace common {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T, typename QPClass>
void
exposeDenseQP(nanobind::module_& m)
{

  ::nanobind::class_<QPClass>(m, "QP")
    .def(::nanobind::init<isize,
                          isize,
                          isize,
                          bool,
                          proxsuite::proxqp::HessianType,
                          proxsuite::proxqp::DenseBackend>(),
         nanobind::arg("n") = 0,
         nanobind::arg("n_eq") = 0,
         nanobind::arg("n_in") = 0,
         nanobind::arg("box_constraints") = false,
         nanobind::arg("hessian_type") = proxsuite::proxqp::HessianType::Dense,
         nanobind::arg("dense_backend") =
           proxsuite::proxqp::DenseBackend::Automatic,
         "Default constructor using QP model dimensions.") // constructor
    .def_rw("results",
            &QPClass::results,
            "class containing the solution or certificate of infeasibility, "
            "and "
            "information statistics in an info subclass.")
    .def_rw("settings", &QPClass::settings, "Settings of the solver.")
    .def_rw("model", &QPClass::model, "class containing the QP model")
    .def("is_box_constrained",
         &QPClass::is_box_constrained,
         "precise whether or not the QP is designed with box constraints.")
    .def("which_hessian_type",
         &QPClass::which_hessian_type,
         "precise which problem type is to be solved.")
    .def("which_dense_backend",
         &QPClass::which_dense_backend,
         "precise which dense backend is chosen.")
    .def("init",
         static_cast<void (QPClass::*)(optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       bool compute_preconditioner,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>)>(&QPClass::init),
         "function for initialize the QP model.",
         nanobind::arg("H"),
         nanobind::arg("g"),
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
    .def("init",
         static_cast<void (QPClass::*)(optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       bool compute_preconditioner,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>)>(&QPClass::init),
         "function for initialize the QP model.",
         nanobind::arg("H") = nanobind::none(),
         nanobind::arg("g") = nanobind::none(),
         nanobind::arg("A") = nanobind::none(),
         nanobind::arg("b") = nanobind::none(),
         nanobind::arg("C") = nanobind::none(),
         nanobind::arg("l") = nanobind::none(),
         nanobind::arg("u") = nanobind::none(),
         nanobind::arg("l_box") = nanobind::none(),
         nanobind::arg("u_box") = nanobind::none(),
         nanobind::arg("compute_preconditioner") = true,
         nanobind::arg("rho") = nanobind::none(),
         nanobind::arg("mu_eq") = nanobind::none(),
         nanobind::arg("mu_in") = nanobind::none(),
         nanobind::arg("manual_minimal_H_eigenvalue") = nanobind::none())
    .def("solve",
         static_cast<void (QPClass::*)()>(&QPClass::solve),
         "function used for solving the QP problem, using default parameters.")
    .def("solve",
         static_cast<void (QPClass::*)(optional<proxqp::dense::VecRef<T>> x,
                                       optional<proxqp::dense::VecRef<T>> y,
                                       optional<proxqp::dense::VecRef<T>> z)>(
           &QPClass::solve),
         "function used for solving the QP problem, when passing a warm start.")

    .def("update",
         static_cast<void (QPClass::*)(optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       bool update_preconditioner,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>)>(&QPClass::update),
         "function used for updating matrix or vector entry of the model using "
         "dense matrix entries.",
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
    .def("update",
         static_cast<void (QPClass::*)(optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::MatRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       optional<proxqp::dense::VecRef<T>>,
                                       bool update_preconditioner,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>,
                                       optional<T>)>(&QPClass::update),
         "function used for updating matrix or vector entry of the model using "
         "dense matrix entries.",
         nanobind::arg("H") = nanobind::none(),
         nanobind::arg("g") = nanobind::none(),
         nanobind::arg("A") = nanobind::none(),
         nanobind::arg("b") = nanobind::none(),
         nanobind::arg("C") = nanobind::none(),
         nanobind::arg("l") = nanobind::none(),
         nanobind::arg("u") = nanobind::none(),
         nanobind::arg("l_box") = nanobind::none(),
         nanobind::arg("u_box") = nanobind::none(),
         nanobind::arg("update_preconditioner") = false,
         nanobind::arg("rho") = nanobind::none(),
         nanobind::arg("mu_eq") = nanobind::none(),
         nanobind::arg("mu_in") = nanobind::none(),
         nanobind::arg("manual_minimal_H_eigenvalue") = nanobind::none())
    .def("cleanup",
         &QPClass::cleanup,
         "function used for cleaning the workspace and result "
         "classes.")
    .def(nanobind::self == nanobind::self)
    .def(nanobind::self != nanobind::self)
    .def("__getstate__",
         [](const QPClass& qp) {
           return proxsuite::serialization::saveToString(qp);
         })
    .def("__setstate__", [](QPClass& qp, const std::string& s) {
      new (&qp) QPClass(1, 1, 1);
      proxsuite::serialization::loadFromString(qp, s);
    });
  ;
}

// TODO: Define some function solveDenseQP with templates or other to avoid code
// duplication (2 functions solveDenseQP in the expose-solve.hpp files)
// Challenge: The functions are not gievn by means of a class like previously,
// by definition of the solve function without API. Then, find some way to
// provide generic coding on namespaces (as wa have
// proxsuite::proxqp::dense::solve<T> and proxsuite::osqp::dense::solve<T>)

} // namespace python
} // namespace dense

} // namespace common
} // namespace proxsuite

#endif // bindings_python_src_common_expose
