#ifndef LDLT_TAGS_HPP_9FTN4PUWS
#define LDLT_TAGS_HPP_9FTN4PUWS

#define LDLT_NOM_SEMICOLON static_assert(true, ".");
#define LDLT_DEFINE_TAG(Name, Type)                                            \
	struct Type {                                                                \
		constexpr explicit Type() noexcept =                                       \
				default; /* should not be an aggergate */                              \
	};                                                                           \
	namespace {                                                                  \
	Type const& Name = ::ldlt::detail::StaticConst<Type>::value; /* NOLINT */    \
	}                                                                            \
	LDLT_NOM_SEMICOLON

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
