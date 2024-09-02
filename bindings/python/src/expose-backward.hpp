//
// Copyright (c) 2023 INRIA
//

#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include "proxsuite/proxqp/dense/compute_ECJ.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
backward(nanobind::module_ m)
{
  m.def("compute_backward",
        &compute_backward<T>,
        "Function for computing derivatives of solved QP.",
        nanobind::arg("qp"),
        nanobind::arg("loss_derivative"),
        nanobind::arg("eps") = 1e-4,
        nanobind::arg("rho_backward") = 1e-6,
        nanobind::arg("mu_backward") = 1e-6);
}

} // namespace python
} // namespace dense

} // namespace proxqp
} // namespace proxsuite
