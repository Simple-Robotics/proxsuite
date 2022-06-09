#ifndef VEG_BYTE_STRING_REF_HPP_NK1AVIX7S
#define VEG_BYTE_STRING_REF_HPP_NK1AVIX7S

#include "veg/type_traits/constructible.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace _detail {
struct ByteStringView {
	VEG_INLINE constexpr ByteStringView(char const* data, isize len)
			VEG_ALWAYS_NOEXCEPT : data_{data},
														len_{len} {}

	ByteStringView(char const* str) VEG_ALWAYS_NOEXCEPT;

	VEG_TEMPLATE(
			typename T,
			requires(
					VEG_CONCEPT(constructible<
											char const*,
											decltype(VEG_DECLVAL(T const&).data())>) &&
					VEG_CONCEPT(
							constructible<isize, decltype(VEG_DECLVAL(T const&).size())>)),

			VEG_INLINE constexpr ByteStringView,
			(arg, T const&))
	VEG_ALWAYS_NOEXCEPT : ByteStringView{
														static_cast<char const*>(arg.data()),
														static_cast<isize>(arg.size())} {}

	char const* data_;
	isize len_;

	VEG_NODISCARD VEG_INLINE constexpr auto data() const VEG_ALWAYS_NOEXCEPT
			-> char const* {
		return data_;
	}
	VEG_NODISCARD VEG_INLINE constexpr auto size() const VEG_ALWAYS_NOEXCEPT
			-> isize {
		return len_;
	}
	VEG_NODISCARD VEG_INLINE auto
	starts_with(ByteStringView other) const VEG_ALWAYS_NOEXCEPT -> bool;
	VEG_NODISCARD VEG_INLINE auto
	operator==(ByteStringView other) const VEG_ALWAYS_NOEXCEPT -> bool;
};
template <typename T>
struct byte_str_static_const {
	static constexpr T value = {"", 0};
};
template <typename T>
constexpr T byte_str_static_const<T>::value;
namespace /* NOLINT */ {
constexpr auto const& empty_str /* NOLINT */ =
		byte_str_static_const<ByteStringView>::value;
} // namespace
} // namespace _detail
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_BYTE_STRING_REF_HPP_NK1AVIX7S */
