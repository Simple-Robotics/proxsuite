//
// Copyright (c) 2022-2024 INRIA
//

#include "pytypedefs.h"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

namespace proxsuite {
namespace common {
namespace python {

namespace detail {
inline auto
type_name_short(nanobind::handle h)
{
  namespace nb = nanobind;
  assert(h.is_type());
  assert(nb::type_check(h));
  return nb::steal<nb::str>(PyType_GetName((PyTypeObject*)h.ptr()));
}
} // namespace detail

/*!
 * Exposes a type and export its values given the first definition in proxqp
 * module.
 *
 * @param m nanobind module (proxsuite).
 */
template<typename E>
void
exposeAndExportValues(nanobind::module_& m, bool export_values = true)
{
  namespace nb = nanobind;
  nb::handle t = nb::type<E>();
  if (!t.is_valid()) {
    throw std::runtime_error("Invalid type");
  }
#ifndef NDEBUG
  assert(t.is_type());
#endif
  nb::enum_<E>& t_ = static_cast<nb::enum_<E>&>(t);
  nb::str name = detail::type_name_short(t);

  m.attr(name) = t_;
  if (export_values) {
    for (nb::handle item : t) {
      m.attr(item.attr("name")) = item;
    }
  }
}

} // namespace python
} // namespace common
} // namespace proxsuite
