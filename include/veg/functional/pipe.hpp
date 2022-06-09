#ifndef VEG_PIPE_HPP_G83ISQINS
#define VEG_PIPE_HPP_G83ISQINS

#include "veg/internal/macros.hpp"
#include "veg/type_traits/invocable.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace fn {
template <typename Fn>
struct Piped {};
} // namespace fn

namespace _detail {
namespace _pipe {
struct PipeableBase {};

VEG_TEMPLATE(
		(typename T, typename Fn, typename Ret = meta::invoke_result_t<Fn, T>),
		requires(VEG_CONCEPT(fn_once<Fn, Ret, T>)),
		VEG_INLINE constexpr auto
		operator|,
		(value, T),
		(fn, Fn&&))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, Ret, T>))
		->meta::invoke_result_t<Fn, T> {
	return VEG_FWD(fn)(VEG_FWD(value));
}
} // namespace _pipe
} // namespace _detail

struct Pipeable : _detail::_pipe::PipeableBase {};
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_PIPE_HPP_G83ISQINS */
