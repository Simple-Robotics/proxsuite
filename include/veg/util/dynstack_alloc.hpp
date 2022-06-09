#ifndef VEG_DYNSTACK_ALLOC_HPP_YYWN4MATS
#define VEG_DYNSTACK_ALLOC_HPP_YYWN4MATS

#include "veg/vec.hpp"
#include "veg/memory/dynamic_stack.hpp"

#define __VEG_IMPL_MAKE_STACK(vec, stack, ...)                                 \
	VEG_NOM_SEMICOLON;                                                           \
	::veg::Vec<unsigned char> vec;                                               \
	vec.resize_for_overwrite((__VA_ARGS__).alloc_req());                         \
	::veg::dynstack::DynStackMut stack{                                          \
			::veg::tags::from_slice_mut, vec.as_mut()};                              \
	VEG_NOM_SEMICOLON

#define VEG_MAKE_STACK(stack, ...)                                             \
	__VEG_IMPL_MAKE_STACK(VEG_ID(stack_storage), stack, __VA_ARGS__)

#endif /* end of include guard VEG_DYNSTACK_ALLOC_HPP_YYWN4MATS */
