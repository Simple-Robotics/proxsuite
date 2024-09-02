//
// Copyright (c) 2022-2024 INRIA
//
#include <proxsuite/fwd.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>

#include "algorithms.hpp"
#include <proxsuite/proxqp/dense/utils.hpp>
#include <proxsuite/helpers/version.hpp>

namespace proxsuite {
namespace proxqp {
namespace python {

template<typename T>
void
exposeCommon(nanobind::module_ m)
{
  exposeResults<T>(m);
  exposeSettings<T>(m);
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  m.def("omp_get_max_threads",
        &omp_get_max_threads,
        "Returns the max number of threads that could be used by OpenMP.");
#endif
}

template<typename T, typename I>
void
exposeSparseAlgorithms(nanobind::module_ m)
{
  sparse::python::exposeSparseModel<T, I>(m);
  sparse::python::exposeQpObjectSparse<T, I>(m);
  sparse::python::exposeQPVectorSparse<T, I>(m);
  sparse::python::solveSparseQp<T, I>(m);
  sparse::python::exposeSparseHelpers<T, I>(m);
}

template<typename T>
void
exposeDenseAlgorithms(nanobind::module_ m)
{
  dense::python::exposeWorkspaceDense<T>(m);
  dense::python::exposeDenseModel<T>(m);
  dense::python::exposeQpObjectDense<T>(m);
  dense::python::exposeQPVectorDense<T>(m);
  dense::python::solveDenseQp<T>(m);
  dense::python::exposeDenseHelpers<T>(m);
}
template<typename T>
void
exposeBackward(nanobind::module_ m)
{
  dense::python::backward<T>(m);
}

#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
template<typename T>
void
exposeDenseParallel(nanobind::module_ m)
{
  dense::python::solveDenseQpParallel<T>(m);
}
template<typename T, typename I>
void
exposeSparseParallel(nanobind::module_ m)
{
  sparse::python::solveSparseQpParallel<T, I>(m);
}
#endif

NB_MODULE(PYTHON_MODULE_NAME, m)
{
  m.doc() = R"pbdoc(
        The proxSuite library
    ------------------------

    .. currentmodule:: proxsuite
    .. autosummary::
        :toctree: _generate

        proxsuite
    )pbdoc";

  nanobind::module_ proxqp_module =
    m.def_submodule("proxqp", "The proxQP solvers of the proxSuite library");
  exposeCommon<f64>(proxqp_module);
  nanobind::module_ dense_module =
    proxqp_module.def_submodule("dense", "Dense solver of proxQP");
  exposeDenseAlgorithms<f64>(dense_module);
  exposeBackward<f64>(dense_module);
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  exposeDenseParallel<f64>(dense_module);
#endif
  nanobind::module_ sparse_module =
    proxqp_module.def_submodule("sparse", "Sparse solver of proxQP");
  exposeSparseAlgorithms<f64, int32_t>(sparse_module);
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  exposeSparseParallel<f64, int32_t>(sparse_module);
#endif

  // Add version
  m.attr("__version__") = helpers::printVersion();

  // Add helpers
  nanobind::module_ helpers_module =
    m.def_submodule("helpers", "Helper module");
  helpers_module.def("printVersion",
                     helpers::printVersion,
                     nanobind::arg("delimiter") = ".",
                     "Print the current version of the package.");
  helpers_module.def("checkVersionAtLeast",
                     helpers::checkVersionAtLeast,
                     nanobind::arg("major_version"),
                     nanobind::arg("minor_version"),
                     nanobind::arg("patch_version"),
                     "Check version of the package is at least greater than "
                     "the one provided as input.");
}

} // namespace python

} // namespace proxqp
} // namespace proxsuite
