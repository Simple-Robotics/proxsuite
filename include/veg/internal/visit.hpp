#ifndef VEG_VISIT_HPP_XPNDWLEHS
#define VEG_VISIT_HPP_XPNDWLEHS

#include "veg/type_traits/invocable.hpp"
#include "veg/type_traits/tags.hpp"
#include "veg/util/unreachable.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace _detail {
template <usize>
struct UTagImpl;

template <usize I>
using UTag = UTagImpl<I>*;

namespace _meta {
template <typename ISeq, typename... Ts>
struct coalesce_impl {
	using Type = void;
};

template <usize I, typename T>
using ignore_I = T;

template <usize... Is, typename T>
struct coalesce_impl<meta::index_sequence<Is...>, ignore_I<Is, T>...> {
	using Type = T;
};
} // namespace _meta
} // namespace _detail
namespace meta {
template <typename... Ts>
struct coalesce
		: _detail::_meta::
					coalesce_impl<meta::make_index_sequence<sizeof...(Ts)>, Ts...> {};

template <typename... Ts>
using coalesce_t = typename coalesce<Ts...>::Type;
} // namespace meta

namespace _detail {
namespace _visit {

template <usize N>
struct visit_impl;

#ifdef __clang__
#define VEG_CLANG_ONLY(...) __VA_ARGS__
#define VEG_NOT_CLANG_ONLY(...)
#else
#define VEG_CLANG_ONLY(...)
#define VEG_NOT_CLANG_ONLY(...) __VA_ARGS__
#endif

template <>
struct visit_impl<0> {
	template <typename R, bool NoExcept, typename Fn>
	[[noreturn]] VEG_INLINE static VEG_CPP14(constexpr) auto apply(
			meta::index_sequence<> /*unused*/, usize /*i*/, Fn /*fn*/) VEG_NOEXCEPT
			-> R {
		meta::unreachable();
	}
};
template <>
struct visit_impl<1> {
	template <typename R, bool NoExcept, typename Fn>
	VEG_INLINE static constexpr auto
	apply(meta::index_sequence<0> /*unused*/, usize i, Fn fn)
			VEG_NOEXCEPT_IF(NoExcept) -> R {
		return (void)i, VEG_CLANG_ONLY((void)meta::unreachable_if(i != 0), )
		                    VEG_FWD(fn)(UTag<0>{});
	}
};
template <>
struct visit_impl<2> {
	template <typename R, bool NoExcept, typename Fn>
	VEG_INLINE static constexpr auto
	apply(meta::index_sequence<0, 1> /*unused*/, usize i, Fn fn)
			VEG_NOEXCEPT_IF(NoExcept) -> R {
		return VEG_CLANG_ONLY((void)meta::unreachable_if(i > 1), )(
				i == 0 ? VEG_FWD(fn)(UTag<0>{}) : VEG_FWD(fn)(UTag<1>{}));
	}
};

#define VEG_DECL_Is(_, Elem) usize __VEG_PP_CAT(I, Elem),
#define VEG_PUT_Is(_, Elem) __VEG_PP_CAT(I, Elem),
#define VEG_CASE_Is(_, Elem)                                                   \
	case __VEG_PP_CAT(I, Elem):                                                  \
		return VEG_FWD(fn)(UTag<__VEG_PP_CAT(I, Elem)>{});

#define VEG_VISIT_IMPL(Tuple)                                                  \
	template <>                                                                  \
	struct visit_impl<__VEG_PP_TUPLE_SIZE(Tuple)> {                              \
		template <                                                                 \
				typename R,                                                            \
				bool NoExcept,                                                         \
				typename Fn,                                                           \
				__VEG_PP_TUPLE_FOR_EACH(VEG_DECL_Is, _, Tuple) usize... Is>            \
		VEG_INLINE static VEG_CPP14(constexpr) auto apply(                         \
                                                                               \
				meta::index_sequence<__VEG_PP_TUPLE_FOR_EACH(VEG_PUT_Is, _, Tuple)     \
		                             Is...> /*tag*/,                               \
				usize i,                                                               \
				Fn fn) VEG_NOEXCEPT_IF(NoExcept) -> R {                                \
                                                                               \
			switch (i) {                                                             \
				__VEG_PP_TUPLE_FOR_EACH(VEG_CASE_Is, _, Tuple)                         \
                                                                               \
			default:                                                                 \
				VEG_NOT_CLANG_ONLY(return VEG_FWD(fn)(UTag<0>{});)                     \
				meta::unreachable();                                                   \
			}                                                                        \
		}                                                                          \
	}

#define VEG_TUPLE __VEG_PP_MAKE_TUPLE(16)
template <usize N>
struct visit_impl {
	template <
			typename R,
			bool NoExcept,
			typename Fn,
			__VEG_PP_TUPLE_FOR_EACH(VEG_DECL_Is, _, VEG_TUPLE) usize... Is>
	VEG_INLINE static VEG_CPP14(constexpr) auto apply(

			meta::index_sequence<__VEG_PP_TUPLE_FOR_EACH(VEG_PUT_Is, _, VEG_TUPLE)
	                             Is...> /*tag*/,
			usize i,
			Fn fn) VEG_NOEXCEPT_IF(NoExcept) -> R {

		switch (i) {
			__VEG_PP_TUPLE_FOR_EACH(VEG_CASE_Is, _, VEG_TUPLE)

		default:
			return visit_impl<N - __VEG_PP_TUPLE_SIZE(VEG_TUPLE)>::
					template apply<R, NoExcept>(
							meta::index_sequence<Is...>{}, i, VEG_FWD(fn));
		}
	}
};

VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(3));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(4));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(5));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(6));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(7));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(8));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(9));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(10));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(11));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(12));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(13));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(14));
VEG_VISIT_IMPL(__VEG_PP_MAKE_TUPLE(15));

#undef VEG_TUPLE
#undef VEG_VISIT_IMPL
#undef VEG_CASE_Is
#undef VEG_PUT_Is
#undef VEG_DECL_Is

template <usize I, usize End>
struct visit11 {
	template <typename Ret, bool NoExcept, typename Fn>
	VEG_INLINE constexpr static auto apply(usize i, Fn fn)
			VEG_NOEXCEPT_IF(NoExcept) -> Ret {
		return i == I ? VEG_FWD(fn)(UTag<I>{})
		              : visit11<I + 1, End>::template apply<Ret, NoExcept>(
												i, VEG_FWD(fn));
	}
};

template <usize End>
struct visit11<End, End> {
	template <typename Ret, bool NoExcept, typename Fn>
	VEG_INLINE static auto apply(usize /*i*/, Fn /*fn*/) VEG_NOEXCEPT -> Ret {
		meta::unreachable();
	}
};
} // namespace _visit

template <typename Ret, bool NoExcept, usize I, typename Fn>
VEG_INLINE VEG_CPP14(constexpr) auto visit14(usize i, Fn fn)
		VEG_NOEXCEPT_IF(NoExcept) -> Ret {
	return _visit::visit_impl<I>::template apply<Ret, NoExcept>(
			meta::make_index_sequence<I>{}, i, VEG_FWD(fn));
}

template <typename Ret, bool NoExcept, usize I, typename Fn>
VEG_INLINE constexpr auto visit11(usize i, Fn fn) VEG_NOEXCEPT_IF(NoExcept)
		-> Ret {
	return _visit::visit11<0, I>::template apply<Ret, NoExcept>(i, VEG_FWD(fn));
}

template <typename Ret, bool NoExcept, usize I, typename Fn>
VEG_INLINE constexpr auto visit(usize i, Fn fn) VEG_NOEXCEPT_IF(NoExcept)
		-> Ret {
#if __cplusplus >= 201402L
	return _detail::visit14<Ret, NoExcept, I>(i, VEG_FWD(fn));
#else
	return meta::is_consteval()
	           ? _detail::visit11<Ret, NoExcept, I>(i, VEG_FWD(fn))
	           : _detail::visit14<Ret, NoExcept, I>(i, VEG_FWD(fn));
#endif
}

} // namespace _detail
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_VISIT_HPP_XPNDWLEHS */
