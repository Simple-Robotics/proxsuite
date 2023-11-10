//
// Copyright (c) 2023 INRIA
//

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include "proxsuite/proxqp/dense/compute_ECJ.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
backward(pybind11::module_ m)
{
  m.def(
    "compute_backward",
    &compute_backward<T>,
    "Function for computing derivatives of solved QP.",
    pybind11::arg_v("qp", "Solved dense QP."),
    pybind11::arg_v("loss_derivative", "Derivate of loss wrt to qp solution."),
    pybind11::arg_v(
      "eps", 1e-4, "Backward pass accuracy for deriving solution Jacobians."),
    pybind11::arg_v("rho_backward",
                    1e-6,
                    "New primal proximal parameter for iterative refinement."),
    pybind11::arg_v("mu_backward",
                    1e-6,
                    "New dual proximal parameter used both for inequality and "
                    "equality for iterative refinement."));
}

} // namespace python
} // namespace dense

} // namespace proxqp
} // namespace proxsuite
