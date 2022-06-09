#ifndef VEG_ASSERT_HPP_VQDAJ2IBS
#define VEG_ASSERT_HPP_VQDAJ2IBS

#include "veg/internal/fix_index.hpp"
#include "veg/internal/typedefs.hpp"
#include "veg/util/compare.hpp"
#include "veg/util/defer.hpp"
#include "veg/internal/byte_string_ref.hpp"
#include "veg/internal/dbg.hpp"
#include "veg/internal/prologue.hpp"
#include <cassert>

#if HEDLEY_HAS_WARNING("-Woverloaded-shift-op-parentheses")
#define __VEG_IGNORE_SHIFT_PAREN_WARNING(Macro, ...)                           \
	HEDLEY_DIAGNOSTIC_PUSH _Pragma(                                              \
			"clang diagnostic ignored \"-Woverloaded-shift-op-parentheses\" ")       \
			Macro(__VA_ARGS__) HEDLEY_DIAGNOSTIC_POP
#else
#define __VEG_IGNORE_SHIFT_PAREN_WARNING(Macro, ...) Macro(__VA_ARGS__)
#endif

#if defined(__GNUC__)

#if defined(__clang__) || __cplusplus >= 201402L
#define __VEG_THIS_FUNCTION __PRETTY_FUNCTION__
#else
#define __VEG_THIS_FUNCTION ""
#endif

#elif defined(HEDLEY_MSVC_VERSION)
#define __VEG_THIS_FUNCTION __FUNCSIG__
#else
#define __VEG_THIS_FUNCTION __func__
#endif

namespace veg {
namespace _detail {
void incr_counter() VEG_ALWAYS_NOEXCEPT;
void decr_counter() VEG_ALWAYS_NOEXCEPT;

[[noreturn]] void on_assert_fail(
		long line, ByteStringView file, ByteStringView func) VEG_NOEXCEPT;
void on_expect_fail(long line, ByteStringView file, ByteStringView func)
		VEG_NOEXCEPT;
void set_assert_params1(ByteStringView op, String lhs, String rhs)
		VEG_ALWAYS_NOEXCEPT;
void set_assert_params2(ByteStringView expression, ByteStringView msg)
		VEG_ALWAYS_NOEXCEPT;

struct cleanup { // NOLINT(cppcoreguidelines-special-member-functions)
	~cleanup() VEG_ALWAYS_NOEXCEPT;
};
} // namespace _detail

namespace _detail {
namespace _assert {

template <typename T>
auto to_string(T const& arg) -> _detail::String {
	_detail::String buf{};

	{
		_detail::incr_counter();
		auto&& cleanup = defer(_detail::decr_counter);
		(void)cleanup;
		veg::fmt::Debug<meta::decay_t<T>>::to_string(buf, ref(arg));
	}

	return buf;
}

template <typename T>
struct lhs_all_of_t {
	T const& lhs;

	template <typename U>
	VEG_INLINE void
	on_assertion_fail(U const& rhs, _detail::ByteStringView op_str) const {
		_detail::set_assert_params1(
				op_str, _assert::to_string(lhs), _assert::to_string(rhs));
	}

#define VEG_DISABLE(Op)                                                        \
	template <typename U>                                                        \
	void operator Op(U const& rhs) = delete

	VEG_DISABLE(&&);
	VEG_DISABLE(||);
	VEG_DISABLE(=);
	VEG_DISABLE(+=);
	VEG_DISABLE(-=);
	VEG_DISABLE(*=);
	VEG_DISABLE(/=);
	VEG_DISABLE(%=);
	VEG_DISABLE(<<=);
	VEG_DISABLE(>>=);
	VEG_DISABLE(&=);
	VEG_DISABLE(^=);
	VEG_DISABLE(|=);
	VEG_DISABLE(<<);
	VEG_DISABLE(>>);
	VEG_DISABLE(&);
	VEG_DISABLE(|);
	VEG_DISABLE(+);
	VEG_DISABLE(-);
	VEG_DISABLE(*);
	VEG_DISABLE(/);
	VEG_DISABLE(%);
	VEG_DISABLE(^);

	// if a VEG_ASSERT fails during the evaluation of the comparison result, the
	// behavior is undefined (potential recursion)

	VEG_TEMPLATE(
			typename U,
			requires(VEG_CONCEPT(eq<T, U>)),
			VEG_INLINE constexpr auto
			operator==,
			(rhs, U const&))
	const->bool {
		return process_op(
				veg::cmp::eq(ref(lhs), ref(rhs)), rhs, {" == ", sizeof(" == ") - 1});
	}
	VEG_TEMPLATE(
			typename U,
			requires(VEG_CONCEPT(eq<T, U>)),
			VEG_INLINE constexpr auto
			operator!=,
			(rhs, U const&))
	const->bool {
		return process_op(
				!(veg::cmp::eq(ref(lhs), ref(rhs))), rhs, {" != ", sizeof(" != ") - 1});
	}

	VEG_TEMPLATE(
			typename U,
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator<,
			(rhs, U const&))
	const->bool {
		return process_op(
				veg::cmp::lt(ref(lhs), ref(rhs)), rhs, {" < ", sizeof(" < ") - 1});
	}
	VEG_TEMPLATE(
			typename U,
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator>,
			(rhs, U const&))
	const->bool {
		return process_op(
				veg::cmp::gt(ref(lhs), ref(rhs)), rhs, {" > ", sizeof(" > ") - 1});
	}
	VEG_TEMPLATE(
			typename U,
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator<=,
			(rhs, U const&))
	const->bool {
		return process_op(
				veg::cmp::leq(ref(lhs), ref(rhs)), rhs, {" <= ", sizeof(" <= ") - 1});
	}
	VEG_TEMPLATE(
			typename U,
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator>=,
			(rhs, U const&))
	const->bool {
		return process_op(
				veg::cmp::geq(ref(lhs), ref(rhs)), rhs, {" >= ", sizeof(" >= ") - 1});
	}

#undef VEG_DISABLE

	VEG_INLINE explicit constexpr
	operator bool() const { // NOLINT(hicpp-explicit-conversions)
		return process_bool(static_cast<bool>(lhs));
	}

	VEG_NODISCARD VEG_INLINE constexpr auto process_bool(bool res) const -> bool {
		return (res ? void(0)
		            : _detail::set_assert_params1(
											_detail::empty_str, _assert::to_string(lhs), {})),
		       res;
	}
	template <typename U>
	VEG_NODISCARD VEG_INLINE constexpr auto
	process_op(bool res, U const& rhs, _detail::ByteStringView op) const -> bool {
		return (res ? void(0) : on_assertion_fail(rhs, op)), res;
	}
};

struct decomposer {
	template <typename T>
	VEG_INLINE constexpr auto
	operator<<(T const& lhs) const VEG_NOEXCEPT->lhs_all_of_t<T> {
		return {lhs};
	}
};

#define __VEG_IMPL_ASSERT_IMPL(Callback, If_Fail, ...)                         \
	static_cast<void>(                                                           \
			(::veg::_detail::_assert::decomposer{} << __VA_ARGS__)                   \
					? (void)(0)                                                          \
					: ((void)(::veg::_detail::cleanup{}),                                \
	           ::veg::_detail::set_assert_params2(                               \
								 ::veg::_detail::ByteStringView{                               \
										 static_cast<char const*>(#__VA_ARGS__),                   \
										 sizeof(#__VA_ARGS__) - 1},                                \
								 If_Fail),                                                     \
	           ::veg::_detail::Callback(                                         \
								 __LINE__,                                                     \
								 ::veg::_detail::ByteStringView{                               \
										 static_cast<char const*>(__FILE__),                       \
										 sizeof(__FILE__) - 1},                                    \
								 ::veg::_detail::ByteStringView{                               \
										 static_cast<char const*>(__VEG_THIS_FUNCTION),            \
										 sizeof(__VEG_THIS_FUNCTION) - 1})))

#define VEG_ASSERT_ELSE(...)                                                   \
	__VEG_IGNORE_SHIFT_PAREN_WARNING(                                            \
			__VEG_IMPL_ASSERT_IMPL, on_assert_fail, __VA_ARGS__)
#define VEG_EXPECT_ELSE(...)                                                   \
	__VEG_IGNORE_SHIFT_PAREN_WARNING(                                            \
			__VEG_IMPL_ASSERT_IMPL, on_expect_fail, __VA_ARGS__)

#define VEG_ASSERT(...) assert((__VA_ARGS__))

#define VEG_EXPECT(...)                                                        \
	__VEG_IGNORE_SHIFT_PAREN_WARNING(                                            \
			__VEG_IMPL_ASSERT_IMPL,                                                  \
			on_expect_fail,                                                          \
			(::veg::_detail::empty_str),                                             \
			__VA_ARGS__)

#define __VEG_IMPL_ALL_OF_1(Ftor, Decomposer, Callback, ...)                   \
	__VEG_IMPL_ALL_OF_2(Ftor, Decomposer, Callback, (__VA_ARGS__))

#define __VEG_IMPL_ASSERT_FTOR_EMPTY(Decomposer, Elem)                         \
	__VEG_IMPL_ASSERT_FTOR(                                                      \
			Decomposer, (::veg::_detail::empty_str, __VEG_PP_REMOVE_PAREN(Elem)))

#define __VEG_IMPL_ASSERT_FTOR(Decomposer, Elem)                               \
	(::veg::_detail::_assert::Decomposer{} << __VEG_PP_TAIL Elem)                \
			? true                                                                   \
			: ((void)(::veg::_detail::cleanup{}),                                    \
	       ::veg::_detail::set_assert_params2(                                   \
						 {reinterpret_cast<char const*>(                                   \
									__VEG_PP_CAT(u8, __VEG_PP_STRINGIZE(__VEG_PP_TAIL Elem))),   \
									sizeof(                                                      \
											__VEG_PP_CAT(                                            \
													u8, __VEG_PP_STRINGIZE(__VEG_PP_TAIL Elem))) -       \
											1},                                                      \
	            __VEG_PP_HEAD Elem),                                             \
						 false),

#define __VEG_IMPL_ALL_OF_2(Ftor, Decomposer, Callback, Tuple)                 \
	(::veg::_detail::all_of({__VEG_PP_TUPLE_FOR_EACH(Ftor, Decomposer, Tuple)})  \
	     ? (void)(0)                                                             \
	     : ::veg::_detail::Callback(                                             \
						 __LINE__,                                                         \
						 {static_cast<char const*>(__FILE__), sizeof(__FILE__) - 1},       \
						 {static_cast<char const*>(__VEG_THIS_FUNCTION),                   \
	            sizeof(__VEG_THIS_FUNCTION) - 1}))

#define VEG_ASSERT_ALL_OF(...) assert(::veg::_detail::all_of({__VA_ARGS__}))

#define VEG_EXPECT_ALL_OF(...)                                                 \
	__VEG_IGNORE_SHIFT_PAREN_WARNING(                                            \
			__VEG_IMPL_ALL_OF_1,                                                     \
			__VEG_IMPL_ASSERT_FTOR_EMPTY,                                            \
			decomposer,                                                              \
			on_expect_fail,                                                          \
			__VA_ARGS__)

#define VEG_ASSERT_ALL_OF_ELSE(...)                                            \
	__VEG_IGNORE_SHIFT_PAREN_WARNING(                                            \
			__VEG_IMPL_ALL_OF_1,                                                     \
			__VEG_IMPL_ASSERT_FTOR,                                                  \
			decomposer,                                                              \
			on_assert_fail,                                                          \
			__VA_ARGS__)

#define VEG_EXPECT_ALL_OF_ELSE(...)                                            \
	__VEG_IGNORE_SHIFT_PAREN_WARNING(                                            \
			__VEG_IMPL_ALL_OF_1,                                                     \
			__VEG_IMPL_ASSERT_FTOR,                                                  \
			decomposer,                                                              \
			on_expect_fail,                                                          \
			__VA_ARGS__)

#define VEG_UNIMPLEMENTED()                                                    \
	VEG_ASSERT(false);                                                           \
	HEDLEY_UNREACHABLE()

} // namespace _assert
} // namespace _detail
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_ASSERT_HPP_VQDAJ2IBS */
