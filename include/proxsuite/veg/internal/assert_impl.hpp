#ifndef VEG_ASSERT_HPP_VQDAJ2IBS
#define VEG_ASSERT_HPP_VQDAJ2IBS

#include "proxsuite/veg/internal/typedefs.hpp"
#include "proxsuite/veg/util/defer.hpp"
#include "proxsuite/veg/internal/dbg.hpp"
#include "proxsuite/veg/internal/prologue.hpp"
#include <cassert>

#define VEG_ASSERT(...) assert((__VA_ARGS__))

#define VEG_ASSERT_ALL_OF(...) assert(::veg::_detail::all_of({__VA_ARGS__}))

#define VEG_UNIMPLEMENTED()                                                    \
	VEG_ASSERT(false);                                                           \
	HEDLEY_UNREACHABLE()

#include "proxsuite/veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_ASSERT_HPP_VQDAJ2IBS */
