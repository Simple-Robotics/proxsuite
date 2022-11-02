//
// Copyright (c) 2022 INRIA
//
#ifndef proxsuite_python_optional_hpp
#define proxsuite_python_optional_hpp

#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <proxsuite/optional.hpp>

template<typename T>
struct pybind11::detail::type_caster<tl::optional<T>>
  : public pybind11::detail::optional_caster<tl::optional<T>>
{
};

template<>
struct pybind11::detail::type_caster<tl::nullopt_t>
  : public pybind11::detail::void_caster<tl::nullopt_t>
{
};

#endif // ifndef proxsuite_python_optional_hpp
