//
// Copyright (c) 2022-2024 INRIA
//

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/operators.h>

#include <proxsuite/proxqp/dense/model.hpp>
#include <proxsuite/proxqp/sparse/model.hpp>
#include <proxsuite/proxqp/dense/utils.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <proxsuite/serialization/model.hpp>

namespace proxsuite {
namespace proxqp {
namespace dense {
namespace python {
template<typename T>
void
exposeDenseModel(nanobind::module_ m)
{

  ::nanobind::class_<BackwardData<T>>(m, "BackwardData")
    .def(::nanobind::init(), "Default constructor.")
    .def(
      "initialize",
      &BackwardData<T>::initialize,
      nanobind::arg("n") = 0,
      nanobind::arg("n_eq") = 0,
      nanobind::arg("n_in") = 0,
      "Initialize the jacobians (allocate memory if not already done) and set"
      " by default their value to zero.")
    .def_ro("dL_dH", &BackwardData<T>::dL_dH)
    .def_ro("dL_dg", &BackwardData<T>::dL_dg)
    .def_ro("dL_dA", &BackwardData<T>::dL_dA)
    .def_ro("dL_db", &BackwardData<T>::dL_db)
    .def_ro("dL_dC", &BackwardData<T>::dL_dC)
    .def_ro("dL_du", &BackwardData<T>::dL_du)
    .def_ro("dL_dl", &BackwardData<T>::dL_dl);
  // .def_ro("dL_dse", &proxsuite::proxqp::dense::BackwardData<T>::dL_dse)
  // .def_ro("dL_dsi",
  // &proxsuite::proxqp::dense::BackwardData<T>::dL_dsi);

  ::nanobind::class_<proxsuite::proxqp::dense::Model<T>>(m, "model")
    .def(::nanobind::init<i64, i64, i64>(),
         nanobind::arg("n") = 0,
         nanobind::arg("n_eq") = 0,
         nanobind::arg("n_in") = 0,
         "Constructor using QP model dimensions.") // constructor)
    .def_ro("H", &Model<T>::H)
    .def_ro("g", &Model<T>::g)
    .def_ro("A", &Model<T>::A)
    .def_ro("b", &Model<T>::b)
    .def_ro("C", &Model<T>::C)
    .def_ro("l", &Model<T>::l)
    .def_ro("u", &Model<T>::u)
    .def_ro("dim", &Model<T>::dim)
    .def_ro("n_eq", &Model<T>::n_eq)
    .def_ro("n_in", &Model<T>::n_in)
    .def_ro("n_total", &Model<T>::n_total)
    .def_rw("backward_data", &Model<T>::backward_data)
    .def("is_valid",
         &Model<T>::is_valid,
         "Check if model is containing valid data.")
    .def(nanobind::self == nanobind::self)
    .def(nanobind::self != nanobind::self)
    .def("__getstate__",
         [](const proxsuite::proxqp::dense::Model<T>& model) {
           return proxsuite::serialization::saveToString(model);
         })
    .def("__setstate__", [](dense::Model<T>& model, const std::string& s) {
      // create qp model which will be updated by loaded data
      new (&model) dense::Model<T>(1, 1, 1);
      proxsuite::serialization::loadFromString(model, s);
    });
}
} // namespace python
} // namespace dense

namespace sparse {
namespace python {
template<typename T, typename I>
void
exposeSparseModel(nanobind::module_ m)
{
  ::nanobind::class_<proxsuite::proxqp::sparse::Model<T, I>>(m, "model")
    .def(::nanobind::init<i64, i64, i64>(),
         nanobind::arg("n") = 0,
         nanobind::arg("n_eq") = 0,
         nanobind::arg("n_in") = 0,
         "Constructor using QP model dimensions.") // constructor)
    .def_ro("g", &Model<T, I>::g)
    .def_ro("b", &Model<T, I>::b)
    .def_ro("l", &Model<T, I>::l)
    .def_ro("u", &Model<T, I>::u)
    .def_ro("dim", &Model<T, I>::dim)
    .def_ro("n_eq", &Model<T, I>::n_eq)
    .def_ro("n_in", &Model<T, I>::n_in)
    .def_ro("H_nnz", &Model<T, I>::H_nnz)
    .def_ro("A_nnz", &Model<T, I>::A_nnz)
    .def_ro("C_nnz", &Model<T, I>::C_nnz);
}
} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
