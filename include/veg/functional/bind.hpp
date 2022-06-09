#ifndef VEG_CURRY_HPP_KV8QZWDES
#define VEG_CURRY_HPP_KV8QZWDES

#include "veg/type_traits/invocable.hpp"
#include "veg/type_traits/constructible.hpp"
#include "veg/tuple.hpp"
#include "veg/functional/pipe.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
template <typename T, usize I>
using inner_ith = decltype(VEG_DECLVAL(T)[Fix<isize{I}>{}]);

namespace _detail {
namespace _fn {

template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret = meta::invoke_result_t<Fn, Args..., StoredArgs...>>
VEG_INLINE static constexpr auto call_bound_back_once(
		Fn&& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...>&& stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(
				VEG_CONCEPT(nothrow_fn_once<Fn, Ret, Args..., StoredArgs...>)) -> Ret {
	return VEG_FWD(fn)(
			VEG_FWD(args)..., __VEG_IMPL_LEAF_ONCE(stored, Is, StoredArgs)...);
}
template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret = meta::invoke_result_t<Fn, StoredArgs..., Args...>>
VEG_INLINE static constexpr auto call_bound_front_once(
		Fn&& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...>&& stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(
				VEG_CONCEPT(nothrow_fn_once<Fn, Ret, StoredArgs..., Args...>)) -> Ret {
	return VEG_FWD(fn)(
			__VEG_IMPL_LEAF_ONCE(stored, Is, StoredArgs)..., VEG_FWD(args)...);
}

template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret = meta::invoke_result_t<Fn&, Args..., RefMut<StoredArgs>...>>
VEG_INLINE static constexpr auto call_bound_back_mut(
		Fn& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...>& stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(VEG_CONCEPT(
				nothrow_fn_mut<Fn, Ret, Args..., RefMut<StoredArgs>...>)) -> Ret {
	return fn(
			VEG_FWD(args)..., mut(__VEG_IMPL_LEAF_MUT(stored, Is, StoredArgs))...);
}
template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret = meta::invoke_result_t<Fn&, RefMut<StoredArgs>..., Args...>>
VEG_INLINE static constexpr auto call_bound_front_mut(
		Fn& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...>& stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(VEG_CONCEPT(
				nothrow_fn_mut<Fn, Ret, RefMut<StoredArgs>..., Args...>)) -> Ret {
	return fn(
			mut(__VEG_IMPL_LEAF_MUT(stored, Is, StoredArgs))..., VEG_FWD(args)...);
}

template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret =
				meta::invoke_result_t<Fn const&, Args..., Ref<StoredArgs>...>>
VEG_INLINE static constexpr auto call_bound_back(
		Fn const& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...> const&
				stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(
				VEG_CONCEPT(nothrow_fn<Fn, Ret, Args..., Ref<StoredArgs>...>)) -> Ret {
	return fn(VEG_FWD(args)..., ref(__VEG_IMPL_LEAF(stored, Is, StoredArgs))...);
}
template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret =
				meta::invoke_result_t<Fn const&, Ref<StoredArgs>..., Args...>>
VEG_INLINE static constexpr auto call_bound_front(
		Fn const& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...> const&
				stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(
				VEG_CONCEPT(nothrow_fn<Fn, Ret, Ref<StoredArgs>..., Args...>)) -> Ret {
	return fn(ref(__VEG_IMPL_LEAF(stored, Is, StoredArgs))..., VEG_FWD(args)...);
}

template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret = meta::invoke_result_t<Fn const&, Args..., StoredArgs...>>
VEG_INLINE static constexpr auto call_bound_back_copy(
		Fn const& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...> const&
				stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn<Fn, Ret, Args..., StoredArgs...>))
				-> Ret {
	return fn(
			VEG_FWD(args)...,
			static_cast<StoredArgs>(__VEG_IMPL_LEAF(stored, Is, StoredArgs))...);
}
template <
		typename Fn,
		typename... StoredArgs,
		typename... Args,
		usize... Is,
		typename Ret = meta::invoke_result_t<Fn const&, StoredArgs..., Args...>>
VEG_INLINE static constexpr auto call_bound_front_copy(
		Fn const& fn,
		tuple::IndexedTuple<meta::index_sequence<Is...>, StoredArgs...> const&
				stored,
		Args&&... args)
		VEG_NOEXCEPT_IF(
				VEG_CONCEPT(nothrow_fn<Fn, Ret, Ref<StoredArgs>..., Args...>)) -> Ret {
	return fn(
			static_cast<StoredArgs>(__VEG_IMPL_LEAF(stored, Is, StoredArgs))...,
			VEG_FWD(args)...);
}

} // namespace _fn
} // namespace _detail

namespace fn {

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindBackOnce {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret = meta::invoke_result_t<Fn, Args..., StoredArgs...>),
			requires(VEG_CONCEPT(fn_once<Fn, Ret, Args..., StoredArgs...>)),
			VEG_INLINE VEG_CPP14(constexpr) auto
			operator(),
			(... args, Args&&)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(
					nothrow_fn_once<Fn, Ret, Args..., StoredArgs...>)) -> Ret {
		return _detail::_fn::call_bound_back_once(
				VEG_FWD(fn), VEG_FWD(stored_args), VEG_FWD(args)...);
	}
};

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindFrontOnce {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret = meta::invoke_result_t<Fn, StoredArgs..., Args...>),
			requires(VEG_CONCEPT(fn_once<Fn, Ret, StoredArgs..., Args...>)),
			VEG_INLINE VEG_CPP14(constexpr) auto
			operator(),
			(... args, Args&&)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(
					nothrow_fn_once<Fn, Ret, StoredArgs..., Args...>)) -> Ret {
		return _detail::_fn::call_bound_front_once(
				VEG_FWD(fn), VEG_FWD(stored_args), VEG_FWD(args)...);
	}
};

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindBackMut {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret =
	         meta::invoke_result_t<Fn&, Args..., RefMut<StoredArgs>...>),
			requires(VEG_CONCEPT(fn_mut<Fn, Ret, Args..., RefMut<StoredArgs>...>)),
			VEG_INLINE VEG_CPP14(constexpr) auto
			operator(),
			(... args, Args&&))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn_mut<Fn, Ret, Args..., RefMut<StoredArgs>...>))
			->Ret {
		return _detail::_fn::call_bound_back_mut(fn, stored_args, VEG_FWD(args)...);
	}
};

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindFrontMut {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret =
	         meta::invoke_result_t<Fn&, RefMut<StoredArgs>..., Args...>),
			requires(VEG_CONCEPT(fn_mut<Fn, Ret, RefMut<StoredArgs>..., Args...>)),
			VEG_INLINE VEG_CPP14(constexpr) auto
			operator(),
			(... args, Args&&))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn_mut<Fn, Ret, RefMut<StoredArgs>..., Args...>))
			->Ret {
		return _detail::_fn::call_bound_front_mut(
				fn, stored_args, VEG_FWD(args)...);
	}
};

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindBack {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret =
	         meta::invoke_result_t<Fn const&, Args..., Ref<StoredArgs>...>),
			requires(VEG_CONCEPT(fn<Fn, Ret, Args..., Ref<StoredArgs>...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn<Fn, Ret, Args..., Ref<StoredArgs>...>))
			->Ret {
		return _detail::_fn::call_bound_back(fn, stored_args, VEG_FWD(args)...);
	}
};

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindFront {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret =
	         meta::invoke_result_t<Fn const&, Ref<StoredArgs>..., Args...>),
			requires(VEG_CONCEPT(fn<Fn, Ret, Ref<StoredArgs>..., Args...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn<Fn, Ret, Ref<StoredArgs>..., Args...>))
			->Ret {
		return _detail::_fn::call_bound_front(fn, stored_args, VEG_FWD(args)...);
	}
};

template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindBackCopy {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret = meta::invoke_result_t<Fn const&, Args..., StoredArgs...>),
			requires(VEG_CONCEPT(fn<Fn, Ret, Args..., StoredArgs...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn<Fn, Ret, Args..., StoredArgs...>))
			->Ret {
		return _detail::_fn::call_bound_back_copy(
				fn, stored_args, VEG_FWD(args)...);
	}
};
template <typename PipeTag, typename Fn, typename... StoredArgs>
struct BindFrontCopy {
	Fn fn;
	Tuple<StoredArgs...> stored_args;

	VEG_TEMPLATE(
			(typename... Args,
	     typename Ret = meta::invoke_result_t<Fn const&, StoredArgs..., Args...>),
			requires(VEG_CONCEPT(fn<Fn, Ret, StoredArgs..., Args...>)),
			VEG_INLINE constexpr auto
			operator(),
			(... args, Args&&))
	const VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn<Fn, Ret, StoredArgs..., Args...>))
			->Ret {
		return _detail::_fn::call_bound_front_copy(
				fn, stored_args, VEG_FWD(args)...);
	}
};

namespace nb {
struct bind_back_once {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindBackOnce<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};
struct bind_front_once {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindFrontOnce<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};

struct bind_back_mut {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindBackMut<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};
struct bind_front_mut {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindFrontMut<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};

struct bind_back {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindBack<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};
struct bind_front {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindFront<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};

struct bind_back_copy {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindBackCopy<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};
struct bind_front_copy {
	template <typename Fn, typename... Args>
	VEG_INLINE constexpr auto operator()(Fn fn, Args... args) const
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Fn>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Args>)))
					-> BindFrontCopy<Pipeable, Fn, Args...> {
		return {
				VEG_FWD(fn),
				Tuple<Args...>{
						inplace[tuplify],
						_detail::MoveFn<Args>{VEG_FWD(args)}...,
				},
		};
	}
};
} // namespace nb
VEG_NIEBLOID(bind_back_once);
VEG_NIEBLOID(bind_front_once);
VEG_NIEBLOID(bind_back_mut);
VEG_NIEBLOID(bind_front_mut);
VEG_NIEBLOID(bind_back);
VEG_NIEBLOID(bind_front);
VEG_NIEBLOID(bind_back_copy);
VEG_NIEBLOID(bind_front_copy);
} // namespace fn

namespace cpo {
#define VEG_CPO(Name)                                                          \
	template <typename Fn, typename... Args>                                     \
	struct is_trivially_constructible<Name /* NOLINT */<Fn, Args...>>            \
			: meta::bool_constant<(                                                  \
						is_trivially_constructible<Fn>::value &&                           \
						VEG_ALL_OF(is_trivially_constructible<Args>::value))> {};          \
	template <typename Fn, typename... Args>                                     \
	struct is_trivially_relocatable<Name /* NOLINT */<Fn, Args...>>              \
			: meta::bool_constant<(                                                  \
						is_trivially_relocatable<Fn>::value &&                             \
						VEG_ALL_OF(is_trivially_relocatable<Args>::value))> {};

VEG_CPO(fn::BindBack);
VEG_CPO(fn::BindBackMut);
VEG_CPO(fn::BindBackOnce);
VEG_CPO(fn::BindBackCopy);

VEG_CPO(fn::BindFront);
VEG_CPO(fn::BindFrontMut);
VEG_CPO(fn::BindFrontOnce);
VEG_CPO(fn::BindFrontCopy);

#undef VEG_CPO
} // namespace cpo
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_CURRY_HPP_KV8QZWDES */
