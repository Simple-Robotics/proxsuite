#include "algorithms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace proxsuite {
namespace qp {
namespace python {

template <typename T,typename I>
void exposeAlgorithm(pybind11::module_ m) {
	qp::dense::python::exposeData<T>(m);
	qp::python::exposeResults<T>(m);
	qp::python::exposeSettings<T>(m);
	qp::python::exposeQpObjectDense<T>(m);
    qp::python::exposeQpObjectSparse<T,I>(m);
    //qp::python::solveDenseQp<T>(m);
    //qp::python::solveSparseQp<T,I>(m);
}

PYBIND11_MODULE(prox_suite, m) {
	m.doc() = R"pbdoc(
        proxsuite module
    ------------------------

    .. currentmodule:: prox_suite
    .. autosummary::
        :toctree: _generate

        prox_suite
    )pbdoc";
	exposeAlgorithm<f64,int32_t>(m);
}

} // namespace python

} // namespace qp
} // namespace proxsuite
