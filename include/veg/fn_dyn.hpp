#ifndef VEG_FN_VIEW_HPP_2EMSNQIXS
#define VEG_FN_VIEW_HPP_2EMSNQIXS

#include "veg/memory/alloc.hpp"
#include "veg/type_traits/alloc.hpp"
#include "veg/ref.hpp"
#include "veg/util/assert.hpp"
#include "veg/util/unreachable.hpp"
#include "veg/internal/delete_special_members.hpp"
#include "veg/tuple.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
template <typename T, usize I>
using inner_ith = decltype(VEG_DECLVAL(T)[Fix<isize{I}>{}]);

namespace fn {
template <bool B, typename T>
struct NoThrowIf;
template <typename T>
using NoThrow = NoThrowIf<true, T>;
template <typename T>
using MayThrow = NoThrowIf<false, T>;
} // namespace fn

namespace _detail {
namespace _fn {
template <typename Sig>
struct valid_sig : meta::false_type {};
template <bool B, typename R, typename... Ts>
struct valid_sig<fn::NoThrowIf<B, R(Ts...) const&>> : meta::true_type {};

template <typename Sig>
struct valid_sig_mut : meta::false_type {};
template <bool B, typename R, typename... Ts>
struct valid_sig_mut<fn::NoThrowIf<B, R(Ts...) const&>> : meta::true_type {};
template <bool B, typename R, typename... Ts>
struct valid_sig_mut<fn::NoThrowIf<B, R(Ts...)&>> : meta::true_type {};

template <typename Sig>
struct valid_sig_once : meta::false_type {};
template <bool B, typename R, typename... Ts>
struct valid_sig_once<fn::NoThrowIf<B, R(Ts...) const&>> : meta::true_type {};
template <bool B, typename R, typename... Ts>
struct valid_sig_once<fn::NoThrowIf<B, R(Ts...)&>> : meta::true_type {};
template <bool B, typename R, typename... Ts>
struct valid_sig_once<fn::NoThrowIf<B, R(Ts...) &&>> : meta::true_type {};

template <typename T, typename Sig>
struct invocable_with_sig;

template <typename T, bool B, typename R, typename... Ts>
struct invocable_with_sig<T, fn::NoThrowIf<B, R(Ts...) const&>> {
	static constexpr bool value = VEG_CONCEPT(fn<T, R, Ts...>);
};
template <typename T, bool B, typename R, typename... Ts>
struct invocable_with_sig<T, fn::NoThrowIf<B, R(Ts...)&>> {
	static constexpr bool value = VEG_CONCEPT(fn_mut<T, R, Ts...>);
};
template <typename T, bool B, typename R, typename... Ts>
struct invocable_with_sig<T, fn::NoThrowIf<B, R(Ts...) &&>> {
	static constexpr bool value = VEG_CONCEPT(fn_once<T, R, Ts...>);
};
} // namespace _fn
} // namespace _detail

namespace concepts {
VEG_DEF_CONCEPT((typename Sig), sig, _detail::_fn::valid_sig<Sig>::value);
VEG_DEF_CONCEPT(
		(typename Sig), sig_mut, _detail::_fn::valid_sig_mut<Sig>::value);
VEG_DEF_CONCEPT(
		(typename Sig), sig_once, _detail::_fn::valid_sig_once<Sig>::value);

VEG_DEF_CONCEPT(
		(typename T, typename Sig),
		fn_with_sig,
		_detail::_fn::invocable_with_sig<T const&, Sig>::value);
VEG_DEF_CONCEPT(
		(typename T, typename Sig),
		fn_mut_with_sig,
		_detail::_fn::invocable_with_sig<T&, Sig>::value);
VEG_DEF_CONCEPT(
		(typename T, typename Sig),
		fn_once_with_sig,
		_detail::_fn::invocable_with_sig<T&&, Sig>::value);
} // namespace concepts

namespace _detail {
namespace _fn {
using fn::NoThrowIf;

template <typename T>
VEG_INLINE void deleter(void* data) noexcept {
	static_cast<T*>(data)->~T();
}
template <usize I, typename F, typename R, typename... Ts>
VEG_INLINE constexpr auto caller_i(void const* data, Ts... ts) noexcept(
		VEG_CONCEPT(nothrow_fn_once<inner_ith<F const&, I>, R, Ts...>)) -> R {
	return (*static_cast<F const*>(data))[Fix<isize{I}>{}](VEG_FWD(ts)...);
}
template <usize I, typename F, typename R, typename... Ts>
VEG_INLINE constexpr auto caller_i_mut(void* data, Ts... ts) noexcept(
		VEG_CONCEPT(nothrow_fn_once<inner_ith<F&, I>, R, Ts...>)) -> R {
	return (*static_cast<F*>(data))[Fix<isize{I}>{}](VEG_FWD(ts)...);
}
template <usize I, typename F, typename R, typename... Ts>
VEG_INLINE constexpr auto caller_i_once(void* data, Ts... ts) noexcept(
		VEG_CONCEPT(nothrow_fn_once<inner_ith<F, I>, R, Ts...>)) -> R {
	return static_cast<F&&>(*static_cast<F*>(data))[Fix<isize{I}>{}](
			VEG_FWD(ts)...);
}

template <typename F, typename R, typename... Ts>
VEG_INLINE constexpr auto caller(void const* data, Ts... ts) noexcept(
		VEG_CONCEPT(nothrow_fn<F, R, Ts...>)) -> R {
	return (*static_cast<F const*>(data))(VEG_FWD(ts)...);
}
template <typename F, typename R, typename... Ts>
VEG_INLINE constexpr auto caller_mut(void* data, Ts... ts) noexcept(
		VEG_CONCEPT(nothrow_fn_mut<F, R, Ts...>)) -> R {
	return (*static_cast<F*>(data))(VEG_FWD(ts)...);
}
template <typename F, typename R, typename... Ts>
VEG_INLINE constexpr auto caller_once(void* data, Ts... ts) noexcept(
		VEG_CONCEPT(nothrow_fn_once<F, R, Ts...>)) -> R {
	return static_cast<F&&>(*static_cast<F*>(data))(VEG_FWD(ts)...);
}

template <usize I, typename Sig>
struct VTableLeafI;
template <usize I, typename R, typename... Ts>
struct VTableLeafI<I, R(Ts...) const>;
template <usize I, bool B, typename R, typename... Ts>
struct VTableLeafI<I, NoThrowIf<B, R(Ts...) const>>;
template <usize I, typename R, typename... Ts>
struct VTableLeafI<I, R(Ts...)>;
template <usize I, bool B, typename R, typename... Ts>
struct VTableLeafI<I, NoThrowIf<B, R(Ts...)>>;

template <usize I, bool B, typename R, typename... Ts>
struct VTableLeafI<I, NoThrowIf<B, R(Ts...) const&>> {
	auto (*fn_ptr)(void const*, Ts...) noexcept(B) -> R;

	template <typename F>
	VEG_INLINE constexpr VTableLeafI(
			meta::true_type /*unused*/, Tag<F> /*tag*/) noexcept //
			: fn_ptr{_fn::caller_i<I, F, R, Ts...>} {}
	template <typename F>
	VEG_INLINE constexpr VTableLeafI(
			meta::false_type /*unused*/, Tag<F> /*tag*/) noexcept //
			: fn_ptr{_fn::caller<F, R, Ts...>} {}
};
template <usize I, bool B, typename R, typename... Ts>
struct VTableLeafI<I, NoThrowIf<B, R(Ts...)&>> {
	auto (*fn_ptr)(void*, Ts...) noexcept(B) -> R;
	template <typename F>
	VEG_INLINE constexpr VTableLeafI(
			meta::true_type /*unused*/, Tag<F> /*tag*/) noexcept
			: fn_ptr{_fn::caller_i_mut<I, F, R, Ts...>} {}
	template <typename F>
	VEG_INLINE constexpr VTableLeafI(
			meta::false_type /*unused*/, Tag<F> /*tag*/) noexcept
			: fn_ptr{_fn::caller_mut<F, R, Ts...>} {}
};
template <usize I, bool B, typename R, typename... Ts>
struct VTableLeafI<I, NoThrowIf<B, R(Ts...) &&>> {
	auto (*fn_ptr)(void*, Ts...) noexcept(B) -> R;
	template <typename F>
	VEG_INLINE constexpr VTableLeafI(
			meta::true_type /*unused*/, Tag<F> /*tag*/) noexcept
			: fn_ptr{_fn::caller_i_once<F, R, Ts...>} {}
	template <typename F>
	VEG_INLINE constexpr VTableLeafI(
			meta::false_type /*unused*/, Tag<F> /*tag*/) noexcept
			: fn_ptr{_fn::caller_once<F, R, Ts...>} {}
};

template <typename Seq, typename... Sigs>
struct VTableI;

template <usize... Is, typename... Sigs>
struct VTableI<meta::index_sequence<Is...>, Sigs...>
		: VTableLeafI<Is, Sigs>... {
	void (*deleter_ptr)(void* data) noexcept;
	usize size_of;
	usize align_of;

	template <typename F>
	VEG_INLINE constexpr VTableI(
			meta::true_type /*unused*/, Tag<F> /*tag*/) noexcept
			: VTableLeafI<Is, Sigs>{meta::true_type{}, Tag<F>{}}...,
				deleter_ptr(_fn::deleter<F>),
				size_of(sizeof(F)),
				align_of(alignof(F)) {}

	template <typename F>
	VEG_INLINE constexpr VTableI(
			meta::false_type /*unused*/, Tag<F> /*tag*/) noexcept
			: VTableLeafI<Is, Sigs>{meta::false_type{}, Tag<F>{}}...,
				deleter_ptr(_fn::deleter<F>),
				size_of(sizeof(F)),
				align_of(alignof(F)) {}
};

template <bool B, typename T, typename Seq, typename... Sigs>
struct VTableIObject;

template <bool B, typename T, usize... Is, typename... Sigs>
struct VTableIObject<B, T, meta::index_sequence<Is...>, Sigs...> {
	static constexpr VTableI<meta::index_sequence<Is...>, Sigs...> value{
			meta::bool_constant<B>{}, Tag<T>{}};
};

template <bool B, typename T, usize... Is, typename... Sigs>
constexpr VTableI<meta::index_sequence<Is...>, Sigs...>
		VTableIObject<B, T, meta::index_sequence<Is...>, Sigs...>::value;

template <typename D, usize I, typename Sig>
struct FnCrtp {};

template <typename D, usize I, bool B, typename R, typename... Ts>
struct FnCrtp<D, I, NoThrowIf<B, R(Ts...) const&>> {
	VEG_INLINE auto operator()(Ts... ts) const noexcept(B) -> R {
		auto& self = static_cast<D const&>(*this);
		auto vtable =
				static_cast<VTableLeafI<I, NoThrowIf<B, R(Ts...) const&>> const*>(
						self.vtable());
		VEG_ASSERT(vtable != nullptr);
		return vtable->fn_ptr(self.data_ref().get(), VEG_FWD(ts)...);
	}
};
template <typename D, usize I, bool B, typename R, typename... Ts>
struct FnCrtp<D, I, NoThrowIf<B, R(Ts...)&>> {
	VEG_INLINE auto operator()(Ts... ts) & noexcept(B) -> R {
		auto& self = static_cast<D&>(*this);
		auto vtable = static_cast<VTableLeafI<I, NoThrowIf<B, R(Ts...)&>> const*>(
				self.vtable());
		VEG_ASSERT(vtable != nullptr);
		return vtable->fn_ptr(self.data_mut(unsafe).get(), VEG_FWD(ts)...);
	}
};
template <typename D, usize I, bool B, typename R, typename... Ts>
struct FnCrtp<D, I, NoThrowIf<B, R(Ts...) &&>> {
	VEG_INLINE auto operator()(Ts... ts) && noexcept(B) -> R {
		auto& self = static_cast<D&>(*this);
		auto vtable = static_cast<VTableLeafI<I, NoThrowIf<B, R(Ts...) &&>> const*>(
				self.vtable());
		VEG_ASSERT(vtable != nullptr);
		return vtable->fn_ptr(self.data_mut(unsafe).get(), VEG_FWD(ts)...);
	}
};

template <typename... Ts>
struct Group;

template <>
struct Group<> {};

template <typename T>
struct Group<T> : T {
	using T::operator();
};

template <typename T, typename... Ts>
struct Group<T, Ts...> : T, Group<Ts...> {
	using T::operator();
	using Group<Ts...>::operator();
};

} // namespace _fn
} // namespace _detail
namespace fn {
template <typename Seq, typename A, typename... Sigs>
struct IndexedFnDyn;
template <typename Seq, typename... Sigs>
struct IndexedFnRefDyn;
template <typename Seq, typename... Sigs>
struct IndexedFnMutDyn;

template <usize... Is, typename... Sigs>
struct IndexedFnRefDyn<meta::index_sequence<Is...>, Sigs...>
		: _detail::_fn::Group<_detail::_fn::FnCrtp<
					IndexedFnRefDyn<meta::index_sequence<Is...>, Sigs...>,
					Is,
					Sigs>...> {
	static_assert(VEG_ALL_OF(VEG_CONCEPT(sig<Sigs>)), ".");

	template <isize I>
	auto operator[](Fix<I> /*tag*/) const noexcept -> _detail::_fn::FnCrtp< //
			IndexedFnRefDyn<meta::index_sequence<Is...>, Sigs...>,
			usize{I},
			ith<usize{I}, Sigs...>> const& {
		return *this;
	}

	using VTable = _detail::_fn::VTableI<meta::index_sequence<Is...>, Sigs...>;
	using _detail::_fn::Group<_detail::_fn::FnCrtp<
			IndexedFnRefDyn<meta::index_sequence<Is...>, Sigs...>,
			Is,
			Sigs>...>::operator();

private:
	struct Raw {
		VTable const* vtable;
		void const* data;
	};

	Raw raw = {};

public:
	VEG_INLINE auto vtable_mut(Unsafe /*tag*/) noexcept -> RefMut<VTable const*> {
		return mut(raw.vtable);
	}
	VEG_INLINE auto vtable() const noexcept -> VTable const* {
		return raw.vtable;
	}
	VEG_INLINE auto vtable_ref() const noexcept -> Ref<VTable const*> {
		return ref(raw.vtable);
	}
	VEG_INLINE auto data_mut(Unsafe /*tag*/) noexcept -> RefMut<void const*> {
		return mut(raw.data);
	}
	VEG_INLINE auto data_ref() const noexcept -> Ref<void const*> {
		return ref(raw.data);
	}
	VEG_INLINE auto ptr() const noexcept -> void const* { return raw.data; }

	IndexedFnRefDyn() = default;

	template <int _>
	IndexedFnRefDyn( //
			Unsafe /*unsafe*/,
			_::FromRawParts<_> /*tag*/,
			VTable const* vtable,
			void const* data) VEG_NOEXCEPT : raw{vtable, data} {}

	VEG_TEMPLATE(
			(typename T),
			requires(VEG_ALL_OF(
					VEG_CONCEPT(fn_once_with_sig<inner_ith<T const&, Is>, Sigs>))),
			VEG_INLINE IndexedFnRefDyn,
			(/*tag*/, FromI),
			(t, Ref<T>))
	VEG_NOEXCEPT : IndexedFnRefDyn{
										 unsafe,
										 from_raw_parts,
										 mem::addressof(_detail::_fn::VTableIObject<
																		true,
																		T,
																		meta::index_sequence<Is...>,
																		Sigs...>::value),
										 mem::addressof(t.get()),
								 } {}

	VEG_TEMPLATE(
			(typename T),
			requires(VEG_ALL_OF(VEG_CONCEPT(fn_with_sig<T, Sigs>))),
			VEG_INLINE IndexedFnRefDyn,
			(/*tag*/, From),
			(t, Ref<T>))
	VEG_NOEXCEPT : IndexedFnRefDyn{
										 unsafe,
										 from_raw_parts,
										 mem::addressof(_detail::_fn::VTableIObject<
																		false,
																		T,
																		meta::index_sequence<Is...>,
																		Sigs...>::value),
										 mem::addressof(t.get()),
								 } {}
};

template <usize... Is, typename... Sigs>
struct IndexedFnMutDyn<meta::index_sequence<Is...>, Sigs...>
		: _detail::_fn::Group<_detail::_fn::FnCrtp<
					IndexedFnMutDyn<meta::index_sequence<Is...>, Sigs...>,
					Is,
					Sigs>...> {
	static_assert(VEG_ALL_OF(VEG_CONCEPT(sig_mut<Sigs>)), ".");

	template <isize I>
	auto operator[](Fix<I> /*tag*/) const noexcept -> _detail::_fn::FnCrtp< //
			IndexedFnMutDyn<meta::index_sequence<Is...>, Sigs...>,
			usize{I},
			ith<usize{I}, Sigs...>> const& {
		return *this;
	}
	template <isize I>
	auto operator[](Fix<I> /*tag*/) noexcept -> _detail::_fn::FnCrtp< //
			IndexedFnMutDyn<meta::index_sequence<Is...>, Sigs...>,
			usize{I},
			ith<usize{I}, Sigs...>>& {
		return *this;
	}

	using VTable = _detail::_fn::VTableI<meta::index_sequence<Is...>, Sigs...>;
	using _detail::_fn::Group<_detail::_fn::FnCrtp<
			IndexedFnMutDyn<meta::index_sequence<Is...>, Sigs...>,
			Is,
			Sigs>...>::operator();

private:
	struct Raw {
		VTable const* vtable = {};
		void* data = {};
	};

	Raw raw;

public:
	VEG_INLINE auto vtable_mut(Unsafe /*tag*/) noexcept -> RefMut<VTable const*> {
		return mut(raw.vtable);
	}
	VEG_INLINE auto vtable() const noexcept -> VTable const* {
		return raw.vtable;
	}
	VEG_INLINE auto vtable_ref() const noexcept -> Ref<VTable const*> {
		return ref(raw.vtable);
	}
	VEG_INLINE auto data_mut(Unsafe /*tag*/) noexcept -> RefMut<void*> {
		return mut(raw.data);
	}
	VEG_INLINE auto data_ref() const noexcept -> Ref<void*> {
		return ref(raw.data);
	}
	VEG_INLINE auto ptr() const noexcept -> void const* { return raw.data; }
	VEG_INLINE auto ptr_mut() noexcept -> void* { return raw.data; }

	IndexedFnMutDyn() = default;

	template <int _>
	IndexedFnMutDyn( //
			Unsafe /*unsafe*/,
			_::FromRawParts<_> /*tag*/,
			VTable const* vtable,
			void* data) VEG_NOEXCEPT : raw{vtable, data} {}

	VEG_TEMPLATE(
			(typename T),
			requires(
					VEG_ALL_OF(VEG_CONCEPT(fn_once_with_sig<inner_ith<T&, Is>, Sigs>))),
			VEG_INLINE IndexedFnMutDyn,
			(/*tag*/, FromI),
			(t, RefMut<T>))
	VEG_NOEXCEPT : IndexedFnMutDyn{
										 unsafe,
										 from_raw_parts,
										 mem::addressof(_detail::_fn::VTableIObject<
																		true,
																		T,
																		meta::index_sequence<Is...>,
																		Sigs...>::value),
										 mem::addressof(t.get()),
								 } {}
	VEG_TEMPLATE(
			(typename T),
			requires(VEG_ALL_OF(VEG_CONCEPT(fn_with_sig<T, Sigs>))),
			VEG_INLINE IndexedFnMutDyn,
			(/*tag*/, From),
			(t, Ref<T>))
	VEG_NOEXCEPT : IndexedFnMutDyn{
										 unsafe,
										 from_raw_parts,
										 mem::addressof(_detail::_fn::VTableIObject<
																		false,
																		T,
																		meta::index_sequence<Is...>,
																		Sigs...>::value),
										 mem::addressof(t.get()),
								 } {}
};

template <usize... Is, typename A, typename... Sigs>
struct IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>
		: _detail::_fn::Group<_detail::_fn::FnCrtp<
					IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>,
					Is,
					Sigs>...> {

	template <isize I>
	auto operator[](Fix<I> /*tag*/) const& noexcept -> _detail::_fn::FnCrtp< //
			IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>,
			usize{I},
			ith<usize{I}, Sigs...>> const& {
		return *this;
	}
	template <isize I>
	auto operator[](Fix<I> /*tag*/) & noexcept -> _detail::_fn::FnCrtp< //
			IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>,
			usize{I},
			ith<usize{I}, Sigs...>>& {
		return *this;
	}
	template <isize I>
	auto operator[](Fix<I> /*tag*/) && noexcept -> _detail::_fn::FnCrtp< //
			IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>,
			usize{I},
			ith<usize{I}, Sigs...>>&& {
		return static_cast<_detail::_fn::FnCrtp< //
				IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>,
				usize{I},
				ith<usize{I}, Sigs...>>&&>(*this);
	}

	static_assert(VEG_ALL_OF(VEG_CONCEPT(sig_once<Sigs>)), ".");
	static_assert(VEG_CONCEPT(nothrow_movable<A>), ".");
	static_assert(VEG_CONCEPT(nothrow_move_assignable<A>), ".");

	using VTable = _detail::_fn::VTableI<meta::index_sequence<Is...>, Sigs...>;
	using _detail::_fn::Group<_detail::_fn::FnCrtp<
			IndexedFnDyn<meta::index_sequence<Is...>, A, Sigs...>,
			Is,
			Sigs>...>::operator();

	struct _raw0 {
		VTable const* vtable = {};
		void* data = {};
	};

	using Raw = Tuple<A, _raw0>;

private:
	Raw raw;

public:
	VEG_INLINE auto alloc_mut(Unsafe /*tag*/) noexcept -> RefMut<A> {
		return mut(raw[0_c]);
	}
	VEG_INLINE auto alloc_ref() const noexcept -> Ref<A> { return ref(raw[0_c]); }
	VEG_INLINE auto vtable_mut(Unsafe /*tag*/) noexcept -> RefMut<VTable const*> {
		return mut(raw[1_c].vtable);
	}
	VEG_INLINE auto vtable() const noexcept -> VTable const* {
		return raw[1_c].vtable;
	}
	VEG_INLINE auto vtable_ref() const noexcept -> Ref<VTable const*> {
		return ref(raw[1_c].vtable);
	}
	VEG_INLINE auto data_mut(Unsafe /*tag*/) noexcept -> RefMut<void*> {
		return mut(raw[1_c].data);
	}
	VEG_INLINE auto data_ref() const noexcept -> Ref<void*> {
		return ref(raw[1_c].data);
	}
	VEG_INLINE auto ptr() const noexcept -> void const* { return raw[1_c].data; }
	VEG_INLINE auto ptr_mut() noexcept -> void* { return raw[1_c].data; }

private:
	void _destroy(void* ptr) noexcept {
		auto vtable = this->vtable();
		vtable->deleter_ptr(ptr);
		mem::Alloc<A>::dealloc(
				alloc_mut(unsafe),
				VEG_FWD(ptr),
				mem::Layout{
						vtable->size_of,
						vtable->align_of,
				});
	}

public:
	IndexedFnDyn() = default;
	VEG_INLINE ~IndexedFnDyn() {
		static_assert(VEG_CONCEPT(alloc::nothrow_dealloc<A>), ".");

		auto ptr = data_ref().get();
		if (ptr != nullptr) {
			_destroy(ptr);
		}
	}
	IndexedFnDyn(IndexedFnDyn const& rhs) noexcept = delete;
	auto operator=(IndexedFnDyn const& rhs) noexcept -> IndexedFnDyn& = delete;
	IndexedFnDyn(IndexedFnDyn&& rhs) noexcept : raw(VEG_FWD(rhs.raw)) {
		raw[1_c] = {};
	}
	auto operator=(IndexedFnDyn&& rhs) noexcept -> IndexedFnDyn& {
		{ auto cleanup = static_cast<IndexedFnDyn>(*this); }
		raw = VEG_FWD(rhs.raw);
		rhs.raw[1_c] = {};
	}

	template <int _>
	VEG_INLINE IndexedFnDyn(
			Unsafe /*unsafe*/,
			_::FromRawParts<_> /*tag*/,
			A alloc,
			VTable const* vtable,
			void* data) VEG_NOEXCEPT : raw{
																		 tuplify,
																		 VEG_FWD(alloc),
																		 _raw0{vtable, data},
																 } {}

	VEG_TEMPLATE(
			(typename FnT, typename T = meta::invoke_result_t<FnT>),
			requires(
					VEG_CONCEPT(fn_once<FnT, T>) &&
					VEG_ALL_OF(VEG_CONCEPT(fn_once_with_sig<inner_ith<T&&, Is>, Sigs>))),
			IndexedFnDyn,
			(/*tag*/, InPlace<FromI>),
			(alloc, A),
			(fn_t, FnT))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_alloc<A>) &&
			VEG_CONCEPT(nothrow_fn_once<FnT, T>))
			: raw{
						inplace[tuplify],
						_detail::MoveFn<A>{VEG_FWD(alloc)},
						_detail::DefaultFn<_raw0>{},
				} {
		auto l = mem::Layout{sizeof(T), alignof(T)};
		_detail::_mem::ManagedAlloc<A> block{
				mem::Alloc<A>::alloc(alloc_mut(unsafe), l).data,
				l,
				this->alloc_mut(unsafe),
		};
		mem::construct_with(static_cast<T*>(block.data), VEG_FWD(fn_t));
		this->data_mut(unsafe).get() = block.data;
		this->vtable_mut(unsafe).get() = mem::addressof(
				_detail::_fn::
						VTableIObject<true, T, meta::index_sequence<Is...>, Sigs...>::
								value);
		block.data = nullptr;
	}

	VEG_TEMPLATE(
			(typename FnT, typename T = meta::invoke_result_t<FnT>),
			requires(
					VEG_CONCEPT(fn_once<FnT, T>) &&
					VEG_ALL_OF(VEG_CONCEPT(fn_once_with_sig<T, Sigs>))),
			IndexedFnDyn,
			(/*tag*/, InPlace<From>),
			(alloc, A),
			(fn_t, FnT))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_alloc<A>) &&
			VEG_CONCEPT(nothrow_fn_once<FnT, T>))
			: raw{
						inplace[tuplify],
						_detail::MoveFn<A>{VEG_FWD(alloc)},
						_detail::DefaultFn<_raw0>{},
				} {
		auto l = mem::Layout{sizeof(T), alignof(T)};
		_detail::_mem::ManagedAlloc<A> block{
				mem::Alloc<A>::alloc(alloc_mut(unsafe), l).data,
				l,
				this->alloc_mut(unsafe),
		};
		mem::construct_with(static_cast<T*>(block.data), VEG_FWD(fn_t));
		this->data_mut(unsafe).get() = block.data;
		this->vtable_mut(unsafe).get() = mem::addressof(
				_detail::_fn::
						VTableIObject<false, T, meta::index_sequence<Is...>, Sigs...>::
								value);
		block.data = nullptr;
	}

	VEG_TEMPLATE(
			(typename T),
			requires(
					VEG_ALL_OF(VEG_CONCEPT(fn_once_with_sig<inner_ith<T&&, Is>, Sigs>))),
			VEG_INLINE IndexedFnDyn,
			(/*tag*/, FromI),
			(alloc, A),
			(t, T))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_alloc<A>) && VEG_CONCEPT(nothrow_movable<T>))
			: IndexedFnDyn{
						inplace[from_i],
						VEG_FWD(alloc),
						_detail::MoveFn<T>{(VEG_FWD(t))}} {}

	VEG_TEMPLATE(
			(typename T),
			requires(VEG_ALL_OF(VEG_CONCEPT(fn_once_with_sig<T, Sigs>))),
			VEG_INLINE IndexedFnDyn,
			(/*tag*/, From),
			(alloc, A),
			(t, T))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_alloc<A>) && VEG_CONCEPT(nothrow_movable<T>))
			: IndexedFnDyn{
						inplace[from], VEG_FWD(alloc), _detail::MoveFn<T>{(VEG_FWD(t))}} {}
};

template <typename... Sigs>
struct FnRefDyn
		: IndexedFnRefDyn<meta::make_index_sequence<sizeof...(Sigs)>, Sigs...> {
	using IndexedFnRefDyn<meta::make_index_sequence<sizeof...(Sigs)>, Sigs...>::
			IndexedFnRefDyn;
};
template <typename... Sigs>
struct FnMutDyn
		: IndexedFnMutDyn<meta::make_index_sequence<sizeof...(Sigs)>, Sigs...> {
	using IndexedFnMutDyn<meta::make_index_sequence<sizeof...(Sigs)>, Sigs...>::
			IndexedFnMutDyn;
};
template <typename A, typename... Sigs>
struct FnDyn
		: IndexedFnDyn<meta::make_index_sequence<sizeof...(Sigs)>, A, Sigs...> {
	using IndexedFnDyn<meta::make_index_sequence<sizeof...(Sigs)>, A, Sigs...>::
			IndexedFnDyn;
};
} // namespace fn
} // namespace veg

#include "veg/internal/epilogue.hpp"

#endif /* end of include guard VEG_FN_VIEW_HPP_2EMSNQIXS */
