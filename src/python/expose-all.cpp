#include "algorithms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/dense/utils.hpp>

namespace qp{

namespace python{

template <typename T>
void exposeAlgorithm(pybind11::module_ m){
    qp::dense::python::exposeData<T>(m);
    //qp::dense::python::exposeWorkspace<T>(m);
    qp::python::exposeResults<T>(m);
    qp::python::exposeSettings<T>(m);
    qp::python::exposeQpObject<T>(m);
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
    exposeAlgorithm<f64>(m);
}


} // namespace python

} // namespace qp