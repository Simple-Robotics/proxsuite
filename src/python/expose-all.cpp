//
// Copyright (c) 2022, INRIA
//
#include "algorithms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace proxsuite {
namespace qp {
namespace python {

template <typename T>
void exposeQpAlgorithm(pybind11::module_ m) {
	qp::python::exposeResults<T>(m);
	qp::python::exposeSettings<T>(m);
}

template <typename T,typename I>
void exposeSparseAlgorithm(pybind11::module_ m) {
    qp::python::exposeQpObjectSparse<T,I>(m);
}

template <typename T>
void exposeDenseAlgorithm(pybind11::module_ m) {
	qp::python::exposeQpObjectDense<T>(m);
}

PYBIND11_MODULE(proxsuite, m) {
	m.doc() = R"pbdoc(
        proxsuite module
    ------------------------

    .. currentmodule:: proxsuite
    .. autosummary::
        :toctree: _generate

        proxsuite
    )pbdoc";

    pybind11::module_ m2 = m.def_submodule("qp", "qp submodule of 'proxsuite' library");
    exposeQpAlgorithm<f64>(m2);
    pybind11::module_ m3 = m2.def_submodule("dense", "dense submodule of 'qp'");
    pybind11::module_ m4 = m2.def_submodule("sparse", "sparse submodule of 'qp'");
	exposeSparseAlgorithm<f64,int32_t>(m4);
    exposeDenseAlgorithm<f64>(m3);
}

} // namespace python

} // namespace qp
} // namespace proxsuite
