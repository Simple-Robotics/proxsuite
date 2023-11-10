//
// Copyright (c) 2022 INRIA
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <proxsuite/proxqp/dense/model.hpp>
#include <proxsuite/proxqp/sparse/model.hpp>
#include <proxsuite/proxqp/dense/utils.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <proxsuite/serialization/model.hpp>
#include "helpers.hpp"
namespace proxsuite {
namespace proxqp {
namespace dense {
namespace python {
template<typename T>
void
exposeDenseModel(pybind11::module_ m)
{

  ::pybind11::class_<proxsuite::proxqp::dense::BackwardData<T>>(
    m, "BackwardData", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.")
    .def(
      "initialize",
      &proxsuite::proxqp::dense::BackwardData<T>::initialize,
      pybind11::arg_v("n", 0, "primal dimension."),
      pybind11::arg_v("n_eq", 0, "number of equality constraints."),
      pybind11::arg_v("n_in", 0, "number of inequality constraints."),
      "Initialize the jacobians (allocate memory if not already done) and set"
      " by default their value to zero.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_dH, "dL_dH.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_dg, "dL_dg.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_dA, "dL_dA.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_db, "dL_db.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_dC, "dL_dC.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_du, "dL_du.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_dl, "dL_dl.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(BackwardData<T>, dL_dl, "dL_dl.");
  // .def_readonly("dL_dH", &proxsuite::proxqp::dense::BackwardData<T>::dL_dH)
  // .def_readonly("dL_dg", &proxsuite::proxqp::dense::BackwardData<T>::dL_dg)
  // .def_readonly("dL_dA", &proxsuite::proxqp::dense::BackwardData<T>::dL_dA)
  // .def_readonly("dL_db",
  //               &proxsuite::proxqp::dense::BackwardData<T>::dL_db)
  // .def_readonly("dL_dC",
  //               &proxsuite::proxqp::dense::BackwardData<T>::dL_dC)
  // .def_readonly("dL_du", &proxsuite::proxqp::dense::BackwardData<T>::dL_du)
  // .def_readonly("dL_dl",
  //               &proxsuite::proxqp::dense::BackwardData<T>::dL_dl)
  // .def_readonly("dL_du", &proxsuite::proxqp::dense::BackwardData<T>::dL_du);
  // .def_readonly("dL_dse", &proxsuite::proxqp::dense::BackwardData<T>::dL_dse)
  // .def_readonly("dL_dsi",
  // &proxsuite::proxqp::dense::BackwardData<T>::dL_dsi);

  ::pybind11::class_<proxsuite::proxqp::dense::Model<T>>(m, "model")
    .def(::pybind11::init<i64, i64, i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         "Constructor using QP model dimensions.") // constructor)
    .def_readonly("H", &Model<T>::H)
    .def_readonly("g", &Model<T>::g)
    .def_readonly("A", &Model<T>::A)
    .def_readonly("b", &Model<T>::b)
    .def_readonly("C", &Model<T>::C)
    .def_readonly("l", &Model<T>::l)
    .def_readonly("u", &Model<T>::u)
    .def_readonly("dim", &Model<T>::dim)
    .def_readonly("n_eq", &Model<T>::n_eq)
    .def_readonly("n_in", &Model<T>::n_in)
    .def_readonly("n_total", &Model<T>::n_total)
    .def_readwrite("backward_data", &Model<T>::backward_data)
    .def("is_valid",
         &Model<T>::is_valid,
         "Check if model is containing valid data.")
    .def(pybind11::self == pybind11::self)
    .def(pybind11::self != pybind11::self)
    .def(pybind11::pickle(

      [](const proxsuite::proxqp::dense::Model<T>& model) {
        return pybind11::bytes(proxsuite::serialization::saveToString(model));
      },
      [](pybind11::bytes& s) {
        // create qp model which will be updated by loaded data
        proxsuite::proxqp::dense::Model<T> model(1, 1, 1);
        proxsuite::serialization::loadFromString(model, s);

        return model;
      }));
}
} // namespace python
} // namespace dense

namespace sparse {
namespace python {
template<typename T, typename I>
void
exposeSparseModel(pybind11::module_ m)
{
  ::pybind11::class_<proxsuite::proxqp::sparse::Model<T, I>>(m, "model")
    .def(::pybind11::init<i64, i64, i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         "Constructor using QP model dimensions.") // constructor)
    .def_readonly("g", &Model<T, I>::g)
    .def_readonly("b", &Model<T, I>::b)
    .def_readonly("l", &Model<T, I>::l)
    .def_readonly("u", &Model<T, I>::u)
    .def_readonly("dim", &Model<T, I>::dim)
    .def_readonly("n_eq", &Model<T, I>::n_eq)
    .def_readonly("n_in", &Model<T, I>::n_in)
    .def_readonly("H_nnz", &Model<T, I>::H_nnz)
    .def_readonly("A_nnz", &Model<T, I>::A_nnz)
    .def_readonly("C_nnz", &Model<T, I>::C_nnz);
}
} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
