#ifndef VEG_TERMINATE_HPP_YMTONE4HS
#define VEG_TERMINATE_HPP_YMTONE4HS

#include "proxsuite/veg/internal/prologue.hpp"
#include <exception>

namespace veg {
namespace _detail {
[[noreturn]] inline void terminate() noexcept {
	std::terminate();
}
} // namespace _detail
} // namespace veg

#include "proxsuite/veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_TERMINATE_HPP_YMTONE4HS */
