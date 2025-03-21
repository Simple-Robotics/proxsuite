//
// Copyright (c) 2022-2024 INRIA
//
#include <proxsuite/fwd.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>

#include "algorithms.hpp"
#include "common/helpers.hpp"
#include <proxsuite/proxqp/dense/utils.hpp>
#include <proxsuite/helpers/version.hpp>

#include "osqp/algorithms.hpp"

namespace proxsuite {
namespace proxqp {
namespace python {

using namespace proxsuite::common::python;
using namespace proxsuite;

template<typename T>
void
exposeCommon(nanobind::module_& m)
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
exposeSparseAlgorithms(nanobind::module_& m)
{
  sparse::python::exposeSparseModel<T, I>(m);
  sparse::python::exposeQpObjectSparse<T, I>(m);
  sparse::python::exposeQPVectorSparse<T, I>(m);
  sparse::python::solveSparseQp<T, I>(m);
  sparse::python::exposeSparseHelpers<T, I>(m);
}

template<typename T>
void
exposeDenseAlgorithms(nanobind::module_& m)
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
exposeBackward(nanobind::module_& m)
{
  dense::python::backward<T>(m);
}

#ifdef PROXSUITE_PYTHON_INTERFACE_WITH_OPENMP
template<typename T>
void
exposeDenseParallel(nanobind::module_& m)
{
  dense::python::solveDenseQpParallel<T>(m);
}
template<typename T, typename I>
void
exposeSparseParallel(nanobind::module_& m)
{
  sparse::python::solveSparseQpParallel<T, I>(m);
}
#endif

/*!
 * Exposes the proxqp module in proxsuite.
 *
 * @param m nanobind module (proxsuite).
 */
void
exposeProxqp(nanobind::module_& m)
{
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
}

/*!
 * Exposes the osqp module in proxsuite from the previous definition of proxqp.
 *
 * @param m nanobind module (proxsuite).
 */
void
exposeOsqp(nanobind::module_& m)
{
  // osqp module
  nanobind::module_ osqp_module =
    m.def_submodule("osqp", "The OSQP solver of the proxSuite library");
  // results
  exposeAndExportValues<QPSolverOutput>(osqp_module);
  osqp_module.attr("Info") = m.attr("proxqp").attr("Info");
  osqp_module.attr("Results") = m.attr("proxqp").attr("Results");
  // settings
  exposeAndExportValues<InitialGuessStatus>(osqp_module);
  exposeAndExportValues<SparseBackend>(osqp_module);
  exposeAndExportValues<EigenValueEstimateMethodOption>(osqp_module);
  osqp_module.attr("Settings") = m.attr("proxqp").attr("Settings");
  // dense module
  nanobind::module_ dense_module =
    osqp_module.def_submodule("dense", "Dense solver of OSQP");
  // workspace
  dense_module.attr("workspace") =
    m.attr("proxqp").attr("dense").attr("workspace");
  // sense model
  dense_module.attr("BackwardData") =
    m.attr("proxqp").attr("dense").attr("BackwardData");
  dense_module.attr("model") = m.attr("proxqp").attr("dense").attr("model");
  // qp object
  exposeAndExportValues<DenseBackend>(dense_module);
  exposeAndExportValues<HessianType>(dense_module);
  osqp::dense::python::exposeQPDense<f64>(dense_module);
  // solve without api
  osqp::dense::python::solveDenseQp<f64>(dense_module);
  // helpers
  dense::python::exposeDenseHelpers<f64>(dense_module);
}

/*!
 * Exposes all the solvers of the library.
 *
 * @param m nanobind module (proxsuite).
 */
void
exposeSolvers(nanobind::module_& m)
{
  exposeProxqp(m);
  exposeOsqp(m);
}

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

  // Expose the solvers
  exposeSolvers(m);

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