//
// Copyright (c) 2022-2024 INRIA
//

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/operators.h>

#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/wrapper.hpp>

#include <proxsuite/osqp/dense/wrapper.hpp>
#include "../common/expose.hpp"

namespace proxsuite {
namespace osqp {
using proxsuite::linalg::veg::isize;

namespace dense {
namespace python {

template<typename T>
void
exposeQPDense(nanobind::module_& m)
{
  proxsuite::common::dense::python::exposeDenseQP<T, dense::QP<T>>(
    m); // (scope: m, name: "QP")
}
} // namespace python
} // namespace dense

} // namespace osqp
} // namespace proxsuite
