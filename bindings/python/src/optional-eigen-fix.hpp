//
// Copyright (c) 2024 INRIA
//
#pragma once

#include <Eigen/Core>
#include <nanobind/stl/optional.h>

#if NB_VERSION_MAJOR < 2

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template<typename T, typename U>
NB_INLINE forwarded_type<T, U>
forward_like_(U&& u)
{
  return (forwarded_type<T, U>)u;
}

template<typename T>
constexpr auto
optional_name(const T& v)
{
  return v + const_name(" | None");
}

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

/// Fix std::optional for Eigen::Ref<const T>
/// Credit to github.com/WKarel for this suggestion!
/// https://github.com/wjakob/nanobind/issues/682#issuecomment-2310746145
template<typename T>
struct type_caster<std::optional<Eigen::Ref<const T>>>
{
  using Ref = Eigen::Ref<const T>;
  using Optional = std::optional<Ref>;
  using Caster = make_caster<Ref>;
  using Map = typename Caster::Map;
  using DMap = typename Caster::DMap;
  NB_TYPE_CASTER(Optional, optional_name(Caster::Name))

  type_caster()
    : value(std::nullopt)
  {
  }

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
  {
    if (src.is_none())
      return true;
    Caster caster;
#if NB_VERSION_MAJOR >= 2
    if (!caster.from_python(src, flags_for_local_caster<T>(flags), cleanup) ||
        !caster.template can_cast<T>())
#else
    if (!caster.from_python(src, flags, cleanup))
#endif
      return false;
    /// This allows us to bypass the type_caster for Eigen::Ref
    /// which is broken due to lack of NRVO + move ctor in latest Eigen release.
    if (caster.dcaster.caster.value.is_valid())
      value.emplace(caster.dcaster.operator DMap());
    else
      value.emplace(caster.caster.operator Map());
    return true;
  }

  template<typename T_>
  static handle from_cpp(T_&& value,
                         rv_policy policy,
                         cleanup_list* cleanup) noexcept
  {
    if (!value)
      return none().release();

    return Caster::from_cpp(forward_like_<T_>(*value), policy, cleanup);
  }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
