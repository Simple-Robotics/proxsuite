#ifndef VEG_UTF8_HPP_B8CJITHGS
#define VEG_UTF8_HPP_B8CJITHGS

#include "veg/internal/macros.hpp"
#include "veg/slice.hpp"

#define VEG_UTF8_CONST(Literal) __VEG_IMPL_UTF8_CONST(Literal)
#define VEG_UTF8(Literal) (__VEG_IMPL_UTF8_CONST(Literal).as_str())

namespace veg {
template <CharUnit... Cs>
constexpr auto StrLiteralConstant<Cs...>::as_slice() const noexcept
		-> Slice<CharUnit> {
	return {unsafe, from_raw_parts, literal, isize{sizeof...(Cs)}};
}
constexpr auto Str::as_slice() const noexcept -> Slice<CharUnit> {
	return {unsafe, from_raw_parts, _.ptr, _.len};
}
} // namespace veg

#endif /* end of include guard VEG_UTF8_HPP_B8CJITHGS */
