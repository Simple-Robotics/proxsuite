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

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {

namespace python {

template<typename T>
void
exposeQpObjectDense(nanobind::module_ m)
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

  // ::nanobind::class_<proxsuite::proxqp::dense::preconditioner::RuizEquilibration<T>>(m,
  // "ruiz")
  //   .def(::nanobind::init(), "Default constructor.")
  //   .def_rw("mu_eq", &RuizEquilibration<T>::delta)
  //   .def_rw("mu_in", &RuizEquilibration<T>::c)
  //   .def_rw("rho", &RuizEquilibration<T>::dim)
  //   .def_rw("iter", &RuizEquilibration<T>::epsilon)
  //   .def_rw("iter_ext", &RuizEquilibration<T>::max_iter)
  //   .def_rw("run_time", &RuizEquilibration<T>::sym);

  // ::nanobind::class_<proxsuite::proxqp::dense::preconditioner::RuizEquilibration<T>>(m,
  // "ruiz")
  //   .def(::nanobind::init(), "Default constructor.")
  //   .def_rw("mu_eq", &RuizEquilibration<T>::delta)
  //   .def_rw("mu_in", &RuizEquilibration<T>::c)
  //   .def_rw("rho", &RuizEquilibration<T>::dim)
  //   .def_rw("iter", &RuizEquilibration<T>::epsilon)
  //   .def_rw("iter_ext", &RuizEquilibration<T>::max_iter)
  //   .def_rw("run_time", &RuizEquilibration<T>::sym);

  ::nanobind::class_<dense::QP<T>>(m, "QP")
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
            &dense::QP<T>::results,
            "class containing the solution or certificate of infeasibility, "
            "and "
            "information statistics in an info subclass.")
    .def_rw("settings", &dense::QP<T>::settings, "Settings of the solver.")
    .def_rw("model", &dense::QP<T>::model, "class containing the QP model")
    .def("is_box_constrained",
         &dense::QP<T>::is_box_constrained,
         "precise whether or not the QP is designed with box constraints.")
    .def("which_hessian_type",
         &dense::QP<T>::which_hessian_type,
         "precise which problem type is to be solved.")
    .def("which_dense_backend",
         &dense::QP<T>::which_dense_backend,
         "precise which dense backend is chosen.")
    .def("init",
         static_cast<void (dense::QP<T>::*)(optional<dense::MatRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::MatRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::MatRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            bool compute_preconditioner,
                                            optional<T>,
                                            optional<T>,
                                            optional<T>,
                                            optional<T>)>(&dense::QP<T>::init),
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
         static_cast<void (dense::QP<T>::*)(optional<dense::MatRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::MatRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::MatRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            optional<dense::VecRef<T>>,
                                            bool compute_preconditioner,
                                            optional<T>,
                                            optional<T>,
                                            optional<T>,
                                            optional<T>)>(&dense::QP<T>::init),
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
         static_cast<void (dense::QP<T>::*)()>(&dense::QP<T>::solve),
         "function used for solving the QP problem, using default parameters.")
    .def("solve",
         static_cast<void (dense::QP<T>::*)(optional<dense::VecRef<T>> x,
                                            optional<dense::VecRef<T>> y,
                                            optional<dense::VecRef<T>> z)>(
           &dense::QP<T>::solve),
         "function used for solving the QP problem, when passing a warm start.")

    .def(
      "update",
      static_cast<void (dense::QP<T>::*)(optional<dense::MatRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::MatRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::MatRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         bool update_preconditioner,
                                         optional<T>,
                                         optional<T>,
                                         optional<T>,
                                         optional<T>)>(&dense::QP<T>::update),
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
    .def(
      "update",
      static_cast<void (dense::QP<T>::*)(optional<dense::MatRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::MatRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::MatRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         optional<dense::VecRef<T>>,
                                         bool update_preconditioner,
                                         optional<T>,
                                         optional<T>,
                                         optional<T>,
                                         optional<T>)>(&dense::QP<T>::update),
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
         &dense::QP<T>::cleanup,
         "function used for cleaning the workspace and result "
         "classes.")
    .def(nanobind::self == nanobind::self)
    .def(nanobind::self != nanobind::self)
    .def("__getstate__",
         [](const dense::QP<T>& qp) {
           return proxsuite::serialization::saveToString(qp);
         })
    .def("__setstate__", [](dense::QP<T>& qp, const std::string& s) {
      new (&qp) dense::QP<T>(1, 1, 1);
      proxsuite::serialization::loadFromString(qp, s);
    });
  ;
}
} // namespace python
} // namespace dense

namespace sparse {

namespace python {

template<typename T, typename I>
void
exposeQpObjectSparse(nanobind::module_ m)
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
