//
// Copyright (c) 2022-2025 INRIA
//

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>

#include "algorithms.hpp"
#include "helpers.hpp"

#include "proxsuite/common/settings.hpp"
#include "proxsuite/common/status.hpp"
#include <proxsuite/common/dense/utils.hpp>

#include <proxsuite/fwd.hpp>
#include <proxsuite/helpers/version.hpp>

namespace proxsuite {
namespace common {
namespace python {

template<typename T, typename I>
void
exposeSparseAlgorithms(nanobind::module_ m)
{
  proxqp::sparse::python::exposeSparseModel<T, I>(m);
  proxqp::sparse::python::exposeQpObjectSparse<T, I>(m);
  proxqp::sparse::python::exposeQPVectorSparse<T, I>(m);
  proxqp::sparse::python::solveSparseQp<T, I>(m);
  proxqp::sparse::python::exposeSparseHelpers<T, I>(m);
}

template<typename T>
void
exposeDenseAlgorithms(nanobind::module_ m)
{
  dense::python::exposeWorkspaceDense<T>(m);
  dense::python::exposeDenseModel<T>(m);
  proxqp::dense::python::exposeQpObjectDense<T>(m);
  proxqp::dense::python::exposeQPVectorDense<T>(m);
  proxqp::dense::python::solveDenseQp<T>(m);
  dense::python::exposeDenseHelpers<T>(m);
}
template<typename T>
void
exposeBackward(nanobind::module_ m)
{
  proxqp::dense::python::backward<T>(m);
}

#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
template<typename T>
void
exposeDenseParallel(nanobind::module_ m)
{
  proxqp::dense::python::solveDenseQpParallel<T>(m);
}
template<typename T, typename I>
void
exposeSparseParallel(nanobind::module_ m)
{
  proxqp::sparse::python::solveSparseQpParallel<T, I>(m);
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

  // PROXQP
  nanobind::module_ proxqp_module =
    m.def_submodule("proxqp", "The proxQP solvers of the proxSuite library");
  exposeResults<f64>(proxqp_module);
  exposeSettings<f64>(proxqp_module);
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  proxqp_module.def(
    "omp_get_max_threads",
    &omp_get_max_threads,
    "Returns the max number of threads that could be used by OpenMP.");
#endif
  nanobind::module_ proxqp_dense_module =
    proxqp_module.def_submodule("dense", "Dense solver of proxQP");
  exposeDenseAlgorithms<f64>(proxqp_dense_module);
  exposeBackward<f64>(proxqp_dense_module);
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  exposeDenseParallel<f64>(dense_module);
#endif
  nanobind::module_ sparse_module =
    proxqp_module.def_submodule("sparse", "Sparse solver of proxQP");
  exposeSparseAlgorithms<f64, int32_t>(sparse_module);
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  exposeSparseParallel<f64, int32_t>(sparse_module);
#endif

  // OSQP
  nanobind::module_ osqp_module =
    m.def_submodule("osqp", "The OSQP solvers of the proxSuite library");
  // exposeResults
  exposeAndExportValues<QPSolverOutput>(osqp_module);
  exposeAndExportValues<PolishStatus>(osqp_module);
  osqp_module.attr("Info") = m.attr("proxqp").attr("Info");
  osqp_module.attr("Results") = m.attr("proxqp").attr("Results");
  // exposeSettings
  exposeAndExportValues<InitialGuessStatus>(osqp_module);
  exposeAndExportValues<SparseBackend>(osqp_module);
  exposeAndExportValues<EigenValueEstimateMethodOption>(osqp_module);
  osqp_module.attr("Settings") = m.attr("proxqp").attr("Settings");
  // OpenMP
#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
  osqp_module.def(
    "omp_get_max_threads",
    &omp_get_max_threads,
    "Returns the max number of threads that could be used by OpenMP.");
#endif
  // dense_module
  nanobind::module_ osqp_dense_module =
    osqp_module.def_submodule("dense", "Dense solver of OSQP");
  // exposeDenseAlgorithms: exposeWorkspaceDense
  osqp_dense_module.attr("workspace") =
    m.attr("proxqp").attr("dense").attr("workspace");
  // exposeDenseAlgorithms: exposeDenseModel
  osqp_dense_module.attr("model") =
    m.attr("proxqp").attr("dense").attr("model");
  // exposeDenseAlgorithms: exposeQpObjectDense
  exposeAndExportValues<DenseBackend>(osqp_dense_module);
  exposeAndExportValues<HessianType>(osqp_dense_module);
  osqp::dense::python::exposeQpObjectDense<f64>(osqp_dense_module);
  // exposeDenseAlgorithms: solveDenseQp
  osqp::dense::python::solveDenseQp<f64>(osqp_dense_module);
  // exposeDenseAlgorithms: exposeDenseHelpers
  osqp_dense_module.attr("estimate_minimal_eigen_value_of_symmetric_matrix") =
    m.attr("proxqp").attr("dense").attr(
      "estimate_minimal_eigen_value_of_symmetric_matrix");

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
} // namespace common
} // namespace proxsuite
