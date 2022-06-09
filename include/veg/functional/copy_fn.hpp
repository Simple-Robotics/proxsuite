#ifndef VEG_COPY_FN_HPP_FY5B8PSTS
#define VEG_COPY_FN_HPP_FY5B8PSTS

#include "veg/type_traits/constructible.hpp"
#include "veg/type_traits/invocable.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace fn {
template <typename Fn>
struct CopyFn {
	VEG_CHECK_CONCEPT(copyable<Fn>);

	Fn fn;
	VEG_TEMPLATE(
			(typename... Args, typename Ret = meta::invoke_result_t<Fn, Args&&...>),
			requires(VEG_CONCEPT(fn_once<Fn, Ret, Args&&...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_copyable<Fn>) &&
			VEG_CONCEPT(nothrow_fn_once<Fn, Ret, Args...>))
			->Ret {
		return Fn(fn)(VEG_FWD(args)...);
	}
};
namespace nb {
struct copy_fn {
	VEG_TEMPLATE(
			typename Fn,
			requires(VEG_CONCEPT(copyable<Fn>)),
			VEG_INLINE constexpr auto
			operator(),
			(fn, Fn))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<Fn>))->CopyFn<Fn> {
		return {Fn(VEG_FWD(fn))};
	}
};
} // namespace nb
VEG_NIEBLOID(copy_fn);
} // namespace fn

namespace cpo {
template <typename Fn>
struct is_trivially_constructible<fn::CopyFn<Fn>>
		: is_trivially_constructible<Fn> {};
template <typename Fn>
struct is_trivially_relocatable<fn::CopyFn<Fn>> : is_trivially_relocatable<Fn> {
};
} // namespace cpo
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_COPY_FN_HPP_FY5B8PSTS */
