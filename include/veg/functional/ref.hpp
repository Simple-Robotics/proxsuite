#ifndef VEG_REF_HPP_5G9FSRMHS
#define VEG_REF_HPP_5G9FSRMHS

#include "veg/ref.hpp"
#include "veg/type_traits/invocable.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace fn {
template <typename Fn>
struct RefFn {
	Ref<Fn> inner;
	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret = meta::invoke_result_t<Fn const&, Args&&...>),
			requires(VEG_CONCEPT(fn<Fn, Ret, Args...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn<Fn, Ret, Args&&...>))->Ret {
		return inner.get()(VEG_FWD(args)...);
	}
};

template <typename Fn>
struct MutFn {
	RefMut<Fn> inner;
	VEG_TEMPLATE(
			(typename... Args, typename Ret = meta::invoke_result_t<Fn&, Args&&...>),
			requires(VEG_CONCEPT(fn_mut<Fn, Ret, Args...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_mut<Fn, Ret, Args&&...>))->Ret {
		return inner.get()(VEG_FWD(args)...);
	}
};

namespace nb {
struct ref {
	template <typename Fn>
	VEG_NODISCARD VEG_INLINE constexpr auto operator()(Ref<Fn> fn) const noexcept
			-> RefFn<Fn> {
		return {fn};
	}
};
struct mut {
	template <typename Fn>
	VEG_NODISCARD VEG_INLINE constexpr auto
	operator()(RefMut<Fn> fn) const noexcept -> MutFn<Fn> {
		return {VEG_FWD(fn)};
	}
};
} // namespace nb
VEG_NIEBLOID(ref);
VEG_NIEBLOID(mut);
} // namespace fn
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_REF_HPP_5G9FSRMHS */
