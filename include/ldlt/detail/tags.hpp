#ifndef LDLT_TAGS_HPP_9FTN4PUWS
#define LDLT_TAGS_HPP_9FTN4PUWS

#include "ldlt/detail/macros.hpp"

#define LDLT_DEFINE_TAG(Name, Type)                                            \
	inline namespace tags {                                                      \
	struct Type {                                                                \
		constexpr explicit Type() noexcept =                                       \
				default; /* should not be an aggergate */                              \
	};                                                                           \
	namespace {                                                                  \
	Type const& Name = ::ldlt::detail::StaticConst<Type>::value; /* NOLINT */    \
	}                                                                            \
	}                                                                            \
	static_assert(sizeof(Name) == 1, ".")

#define LDLT_DEFINE_NIEBLOID(Name)                                             \
	namespace {                                                                  \
	nb::Name const& Name =                                                       \
			::ldlt::detail::StaticConst<nb::Name>::value; /* NOLINT */               \
	}                                                                            \
	static_assert(sizeof(Name) == 1, ".")

namespace ldlt {
namespace detail {
template <typename T>
struct StaticConst {
	static constexpr T value{};
};
template <typename T>
constexpr T StaticConst<T>::value;

template <typename T, typename U>
using DiscardFirst = U;

template <bool B>
struct ConditionalImpl {
	template <typename T, typename F>
	using Type = T;
};
template <>
struct ConditionalImpl<false> {
	template <typename T, typename F>
	using Type = F;
};

template <bool B, typename T, typename F>
using Conditional = typename ConditionalImpl<B>::template Type<T, F>;

template <bool B, typename T>
struct enable_if {
	using type = T;
};

template <typename T>
struct enable_if<false, T> {};

template <bool B, typename T>
using enable_if_t = typename enable_if<B, T>::type;

template <typename T, T V>
struct Constant {
	static constexpr T value = V;
};
template <typename T, T V>
constexpr T Constant<T, V>::value;

template <typename... Preds>
struct Disjunction;
template <typename... Preds>
struct Conjunction;

template <>
struct Disjunction<> : Constant<bool, false> {};
template <>
struct Conjunction<> : Constant<bool, true> {};

template <typename First, typename... Preds>
struct Disjunction<First, Preds...>
		: Conditional<First::value, First, Disjunction<Preds...>> {};

template <typename First, typename... Preds>
struct Conjunction<First, Preds...>
		: Conditional<First::value, Conjunction<Preds...>, First> {};

using True = Constant<bool, true>;
using False = Constant<bool, false>;
} // namespace detail
} // namespace ldlt

#endif /* end of include guard LDLT_TAGS_HPP_9FTN4PUWS */
