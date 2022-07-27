//
// Copyright (c) 2022, INRIA
//
#include "algorithms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <proxsuite/proxqp/dense/utils.hpp>

namespace proxsuite {
namespace proxqp {
namespace python {

template <typename T>
void exposeQpAlgorithms(pybind11::module_ m) {
	exposeResults<T>(m);
	exposeSettings<T>(m);
}

template <typename T,typename I>
void exposeSparseAlgorithms(pybind11::module_ m) {
    sparse::python::exposeSparseModel<T,I>(m);
    sparse::python::exposeQpObjectSparse<T,I>(m);
    sparse::python::solveSparseQp<T,I>(m);
}

template <typename T>
void exposeDenseAlgorithms(pybind11::module_ m) {
	dense::python::exposeDenseModel<T>(m);
    dense::python::exposeQpObjectDense<T>(m);
    dense::python::solveDenseQp<T>(m);
}

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
	m.doc() = R"pbdoc(
        ProxSuite module
    ------------------------

    .. currentmodule:: proxsuite
    .. autosummary::
        :toctree: _generate

        proxsuite
    )pbdoc";

    pybind11::module_ m2 = m.def_submodule("qp", "qp submodule of 'proxsuite' library");
    exposeQpAlgorithms<f64>(m2);
    pybind11::module_ m3 = m2.def_submodule("dense", "dense submodule of 'qp'");
    pybind11::module_ m4 = m2.def_submodule("sparse", "sparse submodule of 'qp'");
	exposeSparseAlgorithms<f64,int32_t>(m4);
    exposeDenseAlgorithms<f64>(m3);
}

} // namespace python

} // namespace proxqp
} // namespace proxsuite
