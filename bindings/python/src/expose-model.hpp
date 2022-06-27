//
// Copyright (c) 2022, INRIA
//
#include <qp/dense/model.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace proxsuite {
namespace qp {
namespace dense {
namespace python {
template <typename T>
void exposeModel(pybind11::module_ m) {
	::pybind11::class_<Model<T>>(m, "Model")
			.def(::pybind11::init<i64, i64, i64>()) // constructor
			.def_readonly("H", &Model<T>::H)
			.def_readonly("g", &Model<T>::g)
			.def_readonly("A", &Model<T>::A)
			.def_readonly("b", &Model<T>::b)
			.def_readonly("C", &Model<T>::C)
			.def_readonly("u", &Model<T>::u)
			.def_readonly("l", &Model<T>::l)
			.def_readonly("dim", &Model<T>::dim)
			.def_readonly("n_eq", &Model<T>::n_eq)
			.def_readonly("n_in", &Model<T>::n_in)
			.def_readonly("n_total", &Model<T>::n_total);
}
} // namespace python
} // namespace dense
} // namespace qp
} // namespace proxsuite
