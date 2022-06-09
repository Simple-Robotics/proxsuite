#ifndef VEG_OVERLOAD_HPP_SRDHJT6LS
#define VEG_OVERLOAD_HPP_SRDHJT6LS

#include "veg/type_traits/constructible.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace fn {
template <typename... Fns>
struct Overload;

#if __cplusplus >= 201703

template <typename... Fns>
struct Overload : Fns... {
	using Fns::operator()...;
	template <typename... FFns>
	constexpr Overload(FFns... ffns)
			VEG_NOEXCEPT_LIKE(VEG_EVAL_ALL(VEG_FWD(ffns)()))
			: Fns{VEG_FWD(ffns)()}... {}
};

#else

template <>
struct Overload<> {
	void operator()() = delete;
};

#define VEG_TYPE_DECL(_, I)                                                    \
	__VEG_PP_COMMA_IF_NOT_0(I) typename __VEG_PP_CAT(Fn, I)
#define VEG_TYPE_PUT(_, I) __VEG_PP_COMMA_IF_NOT_0(I) __VEG_PP_CAT(Fn, I)
#define VEG_TYPE_INHERIT(_, I) __VEG_PP_COMMA_IF_NOT_0(I) __VEG_PP_CAT(Fn, I)

#define VEG_OVERLOAD_CALL(_, I) using __VEG_PP_CAT(Fn, I)::operator();
#define VEG_OVERLOAD_CTOR_DECL_TPL(_, I)                                       \
	__VEG_PP_COMMA_IF_NOT_0(I) typename __VEG_PP_CAT(FFn, I)
#define VEG_OVERLOAD_CTOR_DECL_FN(_, I)                                        \
	__VEG_PP_COMMA_IF_NOT_0(I) __VEG_PP_CAT(FFn, I) __VEG_PP_CAT(ffn, I)
#define VEG_OVERLOAD_CTOR_NOEXCEPT(_, I)                                       \
	__VEG_PP_COMMA_IF_NOT_0(I)(void) VEG_FWD(__VEG_PP_CAT(ffn, I))()
#define VEG_OVERLOAD_CTOR_INIT(_, I)                                           \
	__VEG_PP_COMMA_IF_NOT_0(I)                                                   \
	__VEG_PP_CAT(Fn, I) { VEG_FWD(__VEG_PP_CAT(ffn, I))() }

#define VEG_OVERLOAD_IMPL(Tuple)                                               \
	template <__VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_DECL, _, Tuple)>                  \
	struct Overload<__VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_PUT, _, Tuple)>             \
			: __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_INHERIT, _, Tuple) {                  \
		__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CALL, _, Tuple)                       \
		template <__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_DECL_TPL, _, Tuple)>   \
		VEG_INLINE constexpr Overload(                                             \
				__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_DECL_FN, _, Tuple))          \
				VEG_NOEXCEPT_LIKE(                                                     \
						(__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_NOEXCEPT, _, Tuple)))   \
				: __VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_INIT, _, Tuple) {}         \
	}

#define VEG_OVERLOAD_GENERIC(Tuple)                                            \
	template <__VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_DECL, _, Tuple), typename... Fns> \
	struct Overload<__VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_PUT, _, Tuple), Fns...>     \
			: __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_INHERIT, _, Tuple),                   \
				Overload<Fns...> {                                                     \
		__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CALL, _, Tuple)                       \
		using Overload<Fns...>::operator();                                        \
		template <                                                                 \
				__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_DECL_TPL, _, Tuple),         \
				typename... FFns>                                                      \
		VEG_INLINE constexpr Overload(                                             \
				__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_DECL_FN, _, Tuple),          \
				FFns... ffns)                                                          \
				VEG_NOEXCEPT_LIKE(                                                     \
						(__VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_NOEXCEPT, _, Tuple),    \
		         VEG_EVAL_ALL(VEG_FWD(ffns)())))                                   \
				: __VEG_PP_TUPLE_FOR_EACH(VEG_OVERLOAD_CTOR_INIT, _, Tuple),           \
					Overload<Fns...>{VEG_FWD(ffns)...} {}                                \
	}

VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(1));
VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(2));
VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(3));
VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(4));
VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(5));
VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(6));
VEG_OVERLOAD_IMPL(__VEG_PP_MAKE_TUPLE(7));
VEG_OVERLOAD_GENERIC(__VEG_PP_MAKE_TUPLE(8));

#undef VEG_TYPE_DECL
#undef VEG_TYPE_PUT
#undef VEG_TYPE_INHERIT
#undef VEG_OVERLOAD_CALL
#undef VEG_OVERLOAD_CTOR_DECL_TPL
#undef VEG_OVERLOAD_CTOR_DECL_FN
#undef VEG_OVERLOAD_CTOR_NOEXCEPT
#undef VEG_OVERLOAD_CTOR_INIT
#undef VEG_OVERLOAD_IMPL
#endif

namespace nb {
struct overload {
	template <typename... Fns>
	VEG_INLINE constexpr auto operator()(Fns... fns) const
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Fns>)))
					-> Overload<Fns...> {
		return {_detail::MoveFn<Fns>{VEG_FWD(fns)}...};
	}
};
} // namespace nb
VEG_NIEBLOID(overload);
} // namespace fn

namespace cpo {
template <typename... Fns>
struct is_trivially_constructible<fn::Overload<Fns...>>
		: meta::bool_constant<VEG_ALL_OF(is_trivially_constructible<Fns>::value)> {
};
template <typename... Fns>
struct is_trivially_relocatable<fn::Overload<Fns...>>
		: meta::bool_constant<VEG_ALL_OF(is_trivially_relocatable<Fns>::value)> {};
} // namespace cpo
} // namespace veg

#include "veg/internal/epilogue.hpp"

#endif /* end of include guard VEG_OVERLOAD_HPP_SRDHJT6LS */
