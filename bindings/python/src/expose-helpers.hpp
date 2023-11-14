//
// Copyright (c) 2022 INRIA
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <proxsuite/proxqp/dense/helpers.hpp>
#include <proxsuite/proxqp/sparse/helpers.hpp>

namespace proxsuite {
namespace proxqp {

namespace dense {

namespace python {

template<typename T>
void
exposeDenseHelpers(pybind11::module_ m)
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
    pybind11::arg("H"),
    pybind11::arg_v("estimate_method_option",
                    EigenValueEstimateMethodOption::ExactMethod,
                    "Two options are available for "
                    "estimating smallest eigenvalue: either a power "
                    "iteration algorithm, or an exact method from Eigen."),
    pybind11::arg_v(
      "power_iteration_accuracy", T(1.E-3), "power iteration accuracy."),
    pybind11::arg_v("nb_power_iteration",
                    1000,
                    "maximal number of power iteration executed."));
}
} // namespace python
} // namespace dense

namespace sparse {

namespace python {

template<typename T, typename I>
void
exposeSparseHelpers(pybind11::module_ m)
{
  m.def("estimate_minimal_eigen_value_of_symmetric_matrix",
        &sparse::estimate_minimal_eigen_value_of_symmetric_matrix<T, I>,
        "Function for estimating the minimal eigenvalue of a sparse symmetric "
        "matrix, "
        " using aPower Iteration algorithm (with parameters : "
        "power_iteration_accuracy and nb_power_iteration).",
        pybind11::arg("H"),
        pybind11::arg_v(
          "power_iteration_accuracy", T(1.E-3), "power iteration accuracy."),
        pybind11::arg_v("nb_power_iteration",
                        1000,
                        "maximal number of power iteration executed."));
}

} // namespace python
} // namespace sparse

} // namespace proxqp
} // namespace proxsuite
