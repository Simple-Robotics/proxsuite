//
// Copyright (c) 2022-2024 INRIA
//

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include <proxsuite/proxqp/dense/helpers.hpp>
#include <proxsuite/proxqp/sparse/helpers.hpp>

namespace proxsuite {
namespace proxqp {

namespace dense {

namespace python {

template<typename T>
void
exposeDenseHelpers(nanobind::module_ m)
{
  m.def(
    "estimate_minimal_eigen_value_of_symmetric_matrix",
    +[](const MatRef<T>& H,
        EigenValueEstimateMethodOption estimate_method_option,
        T power_iteration_accuracy,
        isize nb_power_iteration) {
      return dense::estimate_minimal_eigen_value_of_symmetric_matrix(
        H,
        estimate_method_option,
        power_iteration_accuracy,
        nb_power_iteration);
    },
    "Function for estimating the minimal eigenvalue of a dense symmetric "
    "matrix. "
    "Two options are available: an exact method using "
    "SelfAdjointEigenSolver from Eigen, "
    "or a Power Iteration algorithm (with parameters : "
    "power_iteration_accuracy and nb_power_iteration).",
    nanobind::arg("H"),
    nanobind::arg("estimate_method_option") =
      EigenValueEstimateMethodOption::ExactMethod,
    nanobind::arg("power_iteration_accuracy") = T(1.E-3),
    nanobind::arg("nb_power_iteration") = 1000);
}
} // namespace python
} // namespace dense

namespace sparse {

namespace python {

template<typename T, typename I>
void
exposeSparseHelpers(nanobind::module_ m)
{
  m.def("estimate_minimal_eigen_value_of_symmetric_matrix",
        &sparse::estimate_minimal_eigen_value_of_symmetric_matrix<T, I>,
        "Function for estimating the minimal eigenvalue of a sparse symmetric "
        "matrix, "
        " using aPower Iteration algorithm (with parameters : "
        "power_iteration_accuracy and nb_power_iteration).",
        nanobind::arg("H"),
        nanobind::arg("power_iteration_accuracy") = T(1.E-3),
        nanobind::arg("nb_power_iteration") = 1000);
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
