#ifndef VEG_TERMINATE_HPP_YMTONE4HS
#define VEG_TERMINATE_HPP_YMTONE4HS

#include "veg/internal/prologue.hpp"
#include <exception>

namespace veg {
namespace _detail {
[[noreturn]] void terminate() noexcept {
  std::terminate();
}
} // namespace _detail
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_TERMINATE_HPP_YMTONE4HS */
