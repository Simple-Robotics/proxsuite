#ifndef LDLT_TAGS_HPP_9FTN4PUWS
#define LDLT_TAGS_HPP_9FTN4PUWS

#include "ldlt/detail/macros.hpp"

#define LDLT_DEFINE_TAG(Name, Type)                                            \
	struct Type {                                                                \
		constexpr explicit Type() noexcept =                                       \
				default; /* should not be an aggergate */                              \
	};                                                                           \
	namespace {                                                                  \
	Type const& Name = ::ldlt::detail::StaticConst<Type>::value; /* NOLINT */    \
	}                                                                            \
	static_assert(sizeof(Name) == 1, ".");

#define LDLT_DEFINE_NIEBLOID(Name)                                             \
	namespace {                                                                  \
	nb::Name const& Name =                                                       \
			::ldlt::detail::StaticConst<nb::Name>::value; /* NOLINT */               \
	}                                                                            \
	static_assert(sizeof(Name) == 1, ".");

namespace ldlt {
namespace detail {
template <typename T>
struct StaticConst {
	static constexpr T value{};
};
template <typename T>
constexpr T StaticConst<T>::value;
} // namespace detail
} // namespace ldlt

#endif /* end of include guard LDLT_TAGS_HPP_9FTN4PUWS */
