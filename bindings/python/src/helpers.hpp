//
// Copyright (c) 2025 INRIA
//

#include <nanobind/nanobind.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include <string_view>

namespace proxsuite {
namespace common {
namespace python {
namespace detail {
inline nanobind::str
type_name_short(nanobind::handle h)
{
  namespace nb = nanobind;
  assert(h.is_type());
  // nb::type_name return the type_name with modules.
  // In the next step, we will trim the modules to only keep the type name.
  auto str = nb::type_name(h);
  std::string_view work_str(str.c_str());
  auto dot_index = work_str.find_last_of('.');
  if (dot_index == std::string_view::npos) {
    return str;
  } else {
    return nb::str(work_str.data() + dot_index + 1);
  }
  // return nb::str(dot_it++, str.end());
}
} // namespace detail

/*!
 * Exposes a type and export its values given the first definition in
 * a first module. It is motivated by the use of .export_values();
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
