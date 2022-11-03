//
// Copyright (c) 2022 INRIA
//
/**
 * @file optional.hpp
 */

#ifndef PROXSUITE_HELPERS_OPTIONAL_HPP
#define PROXSUITE_HELPERS_OPTIONAL_HPP

#if __cplusplus >= 201703L
#include <optional>
#else
#include <proxsuite/helpers/tl-optional.hpp>
#endif

namespace proxsuite {
#if __cplusplus >= 201703L
template<class T>
using optional = std::optional<T>;
using nullopt_t = std::nullopt_t;
inline constexpr nullopt_t nullopt = std::nullopt;
#else
template<class T>
using optional = tl::optional<T>;
using nullopt_t = tl::nullopt_t;
inline constexpr nullopt_t nullopt = tl::nullopt;
#endif
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_HELPERS_OPTIONAL_HPP */
