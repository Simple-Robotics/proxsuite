//
// Copyright (c) 2022 INRIA
//
/**
 * @file common.hpp
 */

#ifndef PROXSUITE_HELPERS_COMMON_HPP
#define PROXSUITE_HELPERS_COMMON_HPP

#include "proxsuite/config.hpp"
#include <limits>

namespace proxsuite {
namespace helpers {

template<typename Scalar>
struct infinite_bound
{
  static Scalar value()
  {
    using namespace std;
    return sqrt(std::numeric_limits<Scalar>::max());
  }
};

} // helpers
} // proxsuite

#endif // ifndef PROXSUITE_HELPERS_COMMON_HPP
