#ifndef VEG_UWUNION_HPP_OHC4GK5JS
#define VEG_UWUNION_HPP_OHC4GK5JS

#include "veg/ref.hpp"
#include "veg/util/assert.hpp"
#include "veg/memory/address.hpp"
#include "veg/memory/placement.hpp"
#include "veg/internal/delete_special_members.hpp"
#include "veg/type_traits/constructible.hpp"
#include "veg/type_traits/assignable.hpp"
#include "veg/internal/integer_seq.hpp"
#include "veg/internal/visit.hpp"
#include "veg/internal/fix_index.hpp"
#include "veg/internal/dbg.hpp"
#include "veg/cereal/bin_cereal.hpp"

#include "veg/internal/prologue.hpp"

namespace veg {
namespace uwunion {
template <typename Seq, typename... Ts>
struct IndexedUwunion;
}

template <typename T, usize I>
using inner_ith = decltype(VEG_DECLVAL(T)[Fix<isize{I}>{}]);

template <typename... Ts>
struct Uwunion;

namespace _detail {
namespace _uwunion {

struct TrivialTag;
struct NonTrivialTag;

template <bool _, bool TrivialDtor, typename... Ts>
union RawUwunionImpl;

template <typename... Ts>
using RawUwunion = RawUwunionImpl<
		false,
		VEG_ALL_OF(VEG_CONCEPT(trivially_destructible<Ts>)),
		Ts...>;
template <usize I>
struct UwunionGetImpl;

template <bool TrivialDtor>
union RawUwunionImpl<false, TrivialDtor> {
	Empty _{};
};

#define VEG_TYPE_DECL(_, I) , typename __VEG_PP_CAT(T, I)
#define VEG_TYPE_PUT(_, I) , __VEG_PP_CAT(T, I)
#define VEG_UWUNION_HEAD(_, I)                                                 \
	Wrapper<__VEG_PP_CAT(T, I)> __VEG_PP_CAT(head, I);
#define VEG_UWUNION_HEAD_CTOR(_, I)                                            \
	template <usize J, typename Fn>                                              \
	VEG_INLINE constexpr RawUwunionImpl(UTag<I> /*unused*/, UTag<J>, Fn fn)      \
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<J>{}))                                \
			: __VEG_PP_CAT(head, I){                                                 \
						static_cast<decltype(__VEG_PP_CAT(head, I).inner)>(                \
								VEG_FWD(fn)(UTag<J>{}))} {}

#define VEG_UWUNION_SPEC(Tuple)                                                \
	template <bool _ __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_DECL, _, Tuple)>           \
	union RawUwunionImpl<                                                        \
			_,                                                                       \
			true __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_PUT, _, Tuple)> {                  \
		__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD, _, Tuple)                        \
		__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD_CTOR, _, Tuple)                   \
	};                                                                           \
	template <bool _ __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_DECL, _, Tuple)>           \
	union RawUwunionImpl<                                                        \
			_,                                                                       \
			false __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_PUT, _, Tuple)> {                 \
		__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD, _, Tuple)                        \
		__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD_CTOR, _, Tuple)                   \
		VEG_CPP20(constexpr) ~RawUwunionImpl() noexcept {}                         \
	};                                                                           \
	template <>                                                                  \
	struct UwunionGetImpl<__VEG_PP_TUPLE_SIZE(Tuple)> {                          \
		template <typename... Ts>                                                  \
		VEG_INLINE static constexpr auto                                           \
		get(RawUwunion<Ts...> const& u) VEG_NOEXCEPT                               \
				-> Wrapper<ith<__VEG_PP_TUPLE_SIZE(Tuple), Ts...>> const& {            \
			return u.__VEG_PP_CAT(head, __VEG_PP_TUPLE_SIZE(Tuple));                 \
		}                                                                          \
	}

#define VEG_UWUNION_TPL(Tuple)

template <>
struct UwunionGetImpl<0> {
	template <typename... Ts>
	VEG_INLINE static constexpr auto get(RawUwunion<Ts...> const& u) VEG_NOEXCEPT
			-> Wrapper<ith<0, Ts...>> const& {
		return u.head0;
	}
};

VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(1));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(2));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(3));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(4));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(5));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(6));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(7));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(8));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(9));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(10));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(11));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(12));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(13));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(14));
VEG_UWUNION_SPEC(__VEG_PP_MAKE_TUPLE(15));
#define VEG_TUPLE __VEG_PP_MAKE_TUPLE(16)

template <
		bool _ __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_DECL, _, VEG_TUPLE),
		typename... Ts>
union RawUwunionImpl<
		_,
		true __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_PUT, _, VEG_TUPLE),
		Ts...> {
	RawUwunion<Ts...> tail;
	__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD, _, VEG_TUPLE)
	__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD_CTOR, _, VEG_TUPLE)
	template <usize I, usize J, typename Fn>
	VEG_INLINE constexpr RawUwunionImpl(UTag<I> /*unused*/, UTag<J> itag, Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<J>{}))
			: tail{UTag<I - __VEG_PP_TUPLE_SIZE(VEG_TUPLE)>{}, itag, VEG_FWD(fn)} {}
};

template <
		bool _ __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_DECL, _, VEG_TUPLE),
		typename... Ts>
union RawUwunionImpl<
		_,
		false __VEG_PP_TUPLE_FOR_EACH(VEG_TYPE_PUT, _, VEG_TUPLE),
		Ts...> {
	RawUwunion<Ts...> tail;
	__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD, _, VEG_TUPLE)
	__VEG_PP_TUPLE_FOR_EACH(VEG_UWUNION_HEAD_CTOR, _, VEG_TUPLE)
	template <usize I, usize J, typename Fn>
	VEG_INLINE constexpr RawUwunionImpl(UTag<I> /*unused*/, UTag<J> itag, Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<J>{}))
			: tail{UTag<I - __VEG_PP_TUPLE_SIZE(VEG_TUPLE)>{}, itag, VEG_FWD(fn)} {}
	VEG_CPP20(constexpr) ~RawUwunionImpl() noexcept {}
};

template <usize I>
struct UwunionGetImpl {
	template <typename... Ts>
	VEG_INLINE static constexpr auto get(RawUwunion<Ts...> const& u) VEG_NOEXCEPT
			-> Wrapper<ith<I, Ts...>> const& {
		return UwunionGetImpl<I - __VEG_PP_TUPLE_SIZE(VEG_TUPLE)>::get(u.tail);
	}
};

#undef VEG_TUPLE
#undef VEG_UWUNION_TPL
#undef VEG_UWUNION_SPEC
#undef VEG_UWUNION_DEF
#undef VEG_UWUNION_HEAD_CTOR
#undef VEG_UWUNION_HEAD
#undef VEG_TYPE_PUT
#undef VEG_TYPE_DECL

template <bool DoubleStorage, typename ISeq, typename... Ts>
struct NonTrivialUwunionImpl;

template <typename U, typename... Ts>
struct UwunionGetter;

template <typename U, typename... Ts>
struct UwunionGetter<U const&, Ts...> {
	U const& self;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const noexcept
			-> ith<I, Ts...> const& {
		return UwunionGetImpl<I>::get(self).inner;
	}
};

template <typename U, typename... Ts>
struct UwunionGetterRef {
	U const& self;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const noexcept
			-> Ref<ith<I, Ts...>> {
		return ref(UwunionGetImpl<I>::get(self).inner);
	}
};

template <typename U, typename... Ts>
struct UwunionGetterMut {
	U const& self;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const noexcept
			-> RefMut<ith<I, Ts...>> {
		return mut(const_cast<ith<I, Ts...>&>(UwunionGetImpl<I>::get(self).inner));
	}
};

template <typename U, typename... Ts>
struct UwunionGetter<U&, Ts...> {
	U& self;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const noexcept
			-> ith<I, Ts...>& {
		return const_cast<ith<I, Ts...>&>(UwunionGetImpl<I>::get(self).inner);
	}
};
template <typename U, typename... Ts>
struct UwunionGetter<U&&, Ts...> {
	U& self;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const noexcept
			-> ith<I, Ts...>&& {
		return const_cast<ith<I, Ts...>&&>(UwunionGetImpl<I>::get(self).inner);
	}
};

template <typename Fn, typename... Ts>
struct FnMapWrapper {
	Fn&& fn;
	RawUwunion<Ts...> const& ref;

	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*itag*/) const&& //
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(VEG_DECLVAL(ith<I, Ts...>)))
					-> decltype(VEG_FWD(fn)(VEG_DECLVAL(ith<I, Ts...>))) {
		return VEG_FWD(fn)(
				const_cast<ith<I, Ts...>&&>(UwunionGetImpl<I>::get(ref).inner));
	}
};

template <typename Fn, typename... Ts>
struct FnMapIWrapper {
	Fn&& fn;
	RawUwunion<Ts...> const& ref;

	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*itag*/) const&& //
			VEG_NOEXCEPT_LIKE(
					VEG_FWD(fn)[Fix<isize{I}>{}](VEG_DECLVAL(ith<I, Ts...>)))
					-> decltype(VEG_FWD(fn)[Fix<isize{I}>{}](
							VEG_DECLVAL(ith<I, Ts...>))) {
		return VEG_FWD(fn)[Fix<isize{I}>{}](
				const_cast<ith<I, Ts...>&&>(UwunionGetImpl<I>::get(ref).inner));
	}
};

template <typename Ret, typename Fn, typename... Ts>
struct FnVisitWrapper {
	Fn&& fn;
	RawUwunion<Ts...> const& ref;

	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*itag*/) const&& //
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(VEG_DECLVAL(ith<I, Ts...>))) -> Ret {
		return VEG_FWD(fn)(
				const_cast<ith<I, Ts...>&&>(UwunionGetImpl<I>::get(ref).inner));
	}
};

template <typename Ret, typename Fn, typename... Ts>
struct FnVisitIWrapper {
	Fn&& fn;
	RawUwunion<Ts...> const& ref;

	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*itag*/) const&& //
			VEG_NOEXCEPT_LIKE(
					VEG_FWD(fn)[Fix<isize{I}>{}](VEG_DECLVAL(ith<I, Ts...>))) -> Ret {
		return VEG_FWD(fn)[Fix<isize{I}>{}](
				const_cast<ith<I, Ts...>&&>(UwunionGetImpl<I>::get(ref).inner));
	}
};

struct EmplaceTag {};

template <typename Target, typename Fn>
struct EmplaceWrapper {
	Fn const fn;
	usize i;
	template <usize I>
	VEG_INLINE constexpr auto
	operator()(UTag<I> itag) const&& VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{}))
			-> Target {
		return {
				EmplaceTag{},
				itag,
				VEG_FWD(fn),
				i,
		};
	}
};

template <typename U, typename Fn>
struct CtorFn {
	U& self;
	Fn fn;

	template <usize I>
			VEG_INLINE VEG_CPP20(constexpr) void operator()(UTag<I> itag) &&
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(itag)) {
		mem::construct_at(mem::addressof(self), itag, itag, VEG_FWD(fn));
	}
};

template <typename U, typename Fn, typename... Ts>
struct AssignFn {
	U& self;
	Fn fn;

	template <usize I>
			VEG_INLINE VEG_CPP14(constexpr) void operator()(UTag<I> itag) &&
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{})) {
		const_cast<ith<I, Ts...>&>(UwunionGetImpl<I>::get(self).inner) =
				VEG_FWD(fn)(itag);
	}
};

template <bool NoExcept, typename U, typename... Ts>
struct DropFn {
	U& self;

	template <usize I>
	VEG_INLINE VEG_CPP20(constexpr) void
	operator()(UTag<I> /*itag*/) const noexcept {
		mem::destroy_at(mem::addressof(
				const_cast<Wrapper<ith<I, Ts...>>&>(UwunionGetImpl<I>::get(self))));
	}
};

template <typename T>
struct UwunionEmptyRef {
	meta::unref_t<T>* ptr;

	template <typename Fn>
	VEG_INLINE constexpr UwunionEmptyRef(
			EmplaceTag /*etag*/, UTag<0> /*itag*/, Fn fn, usize /*i*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<0>{}))
			: ptr{((void)fn, nullptr)} {}

	template <typename Fn>
	VEG_INLINE constexpr UwunionEmptyRef(
			EmplaceTag /*etag*/, UTag<1> /*itag*/, Fn fn, usize /*i*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<1>{}))
			: ptr{mem::addressof(VEG_FWD(fn)(UTag<1>{}))} {}

	template <usize I, typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void _emplace(Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{})) {
		_emplace2(VEG_FWD(fn), UTag<I>{});
	}

	template <typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void _emplace2(Fn fn, UTag<0> /*itag*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<0>{})) {
		(void)fn;
		ptr = nullptr;
	}
	template <typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void _emplace2(Fn fn, UTag<1> /*itag*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<1>{})) {
		ptr = mem::addressof(VEG_FWD(fn)(UTag<1>{}));
	}

	template <usize I>
			VEG_INLINE constexpr auto get_ref() const & VEG_NOEXCEPT -> T& {
		static_assert(I == 1, ".");
		return *ptr;
	}

	VEG_NODISCARD VEG_INLINE constexpr auto index() const -> isize {
		return isize((ptr == nullptr) ? usize{0} : usize{1});
	}
};

template <typename T>
struct TrivialUwunionImplDefaultCtor {

	T inner;
	bool is_engaged;

	template <typename Fn>
	VEG_INLINE constexpr TrivialUwunionImplDefaultCtor(
			EmplaceTag /*etag*/, UTag<0> /*itag*/, Fn fn, usize /*i*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<0>{}))
			: inner(), is_engaged(((void)fn, false)) {}

	template <typename Fn>
	VEG_INLINE constexpr TrivialUwunionImplDefaultCtor(
			EmplaceTag /*etag*/, UTag<1> /*itag*/, Fn fn, usize /*i*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<1>{}))
			: inner{VEG_FWD(fn)(UTag<1>{})}, is_engaged(true) {}

	template <usize I, typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void _emplace(Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{})) {
		_emplace2(VEG_FWD(fn), UTag<I>{});
	}

	template <typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void _emplace2(Fn fn, UTag<0> /*itag*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<0>{})) {
		(void)fn;
		is_engaged = false;
	}
	template <typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void _emplace2(Fn fn, UTag<1> /*itag*/)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<1>{})) {
		T local = VEG_FWD(fn)(UTag<1>{});
		inner = VEG_FWD(local);
		is_engaged = true;
	}

	template <usize I>
			VEG_INLINE constexpr auto get_ref() const & VEG_NOEXCEPT -> T const& {
		static_assert(I == 1, ".");
		return inner;
	}

	VEG_NODISCARD VEG_INLINE constexpr auto index() const -> isize {
		return isize(is_engaged);
	}
};

template <typename... Ts>
struct TrivialUwunionImplGeneric {
	template <usize I>
	using Ith = ith<I, Ts...>;
	using TagType = meta::if_t<sizeof...(Ts) < 256U, u8, usize>;

	union {
		Empty _;
		RawUwunion<Ts...> inner;
	};
	TagType tag;

	template <usize I, typename Fn>
	VEG_INLINE constexpr TrivialUwunionImplGeneric(
			EmplaceTag /*etag*/, UTag<I> itag, Fn fn, usize i = I)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{}))
			: inner{itag, itag, VEG_FWD(fn)}, tag(TagType(i)) {}

	template <usize I, typename Fn>
	VEG_INLINE VEG_CPP20(constexpr) void _emplace(Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{})) {
		mem::construct_at(mem::addressof(inner), UTag<I>{}, UTag<I>{}, VEG_FWD(fn));
		tag = I;
	}

	template <usize I>
			VEG_INLINE constexpr auto get_ref() const
			& VEG_NOEXCEPT -> Ith<I> const& {
		return (void)meta::unreachable_if(tag != I),
		       UwunionGetImpl<I>::get(inner).inner;
	}

	VEG_NODISCARD VEG_INLINE constexpr auto index() const -> isize {
		return isize(tag);
	}
	VEG_INLINE constexpr auto get_union_ref() const VEG_NOEXCEPT
			-> RawUwunion<Ts...> const& {
		return this->inner;
	}
};

template <typename... Ts>
struct TrivialUwunionImpl : TrivialUwunionImplGeneric<Ts...> {
	using TrivialUwunionImplGeneric<Ts...>::TrivialUwunionImplGeneric;
};

template <typename T>
struct TrivialUwunionImpl<Empty, T&> : UwunionEmptyRef<T&> {
	using UwunionEmptyRef<T&>::UwunionEmptyRef;
};
template <typename T>
struct TrivialUwunionImpl<Empty, T&&> : UwunionEmptyRef<T&&> {
	using UwunionEmptyRef<T&&>::UwunionEmptyRef;
};

template <typename Base, usize N, bool NoExcept, typename Fn>
VEG_INLINE constexpr auto make(Fn fn, usize tag) VEG_NOEXCEPT_IF(NoExcept)
		-> Base {
	return _detail::visit<Base, NoExcept, N>(
			tag, EmplaceWrapper<Base, Fn>{VEG_FWD(fn), tag});
}

template <bool NeedsDtor, typename... Ts>
struct NonTrivialUwunionDtor;

#define VEG_TAGGED_UWUNION_DTOR_true                                           \
	VEG_INLINE VEG_CPP14(constexpr) void destroy() VEG_NOEXCEPT {                \
		_detail::visit<void, true, sizeof...(Ts)>(                                 \
				tag, DropFn<true, decltype(inner), Ts...>{inner});                     \
	}                                                                            \
	VEG_INLINE VEG_CPP20(constexpr) ~NonTrivialUwunionDtor() { destroy(); }

#define VEG_TAGGED_UWUNION_DTOR_false                                          \
	VEG_INLINE VEG_CPP14(constexpr) void destroy() VEG_NOEXCEPT {}

#define VEG_TAGGED_UWUNION_MOVE_false(Class)                                   \
	Class(Class&&) = default; /* NOLINT */
#define VEG_TAGGED_UWUNION_COPY_false(Class) Class(Class const&) = default;

#if __cplusplus >= 201703L

#define VEG_TAGGED_UWUNION_MOVE_true(Class)                                    \
	VEG_INLINE                                                                   \
	constexpr Class(Class&& rhs) /* NOLINT */                                    \
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)))            \
			: inner{_uwunion::make<                                                  \
						decltype(inner),                                                   \
						sizeof...(Ts),                                                     \
						VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>))>(                     \
						UwunionGetter<decltype(inner.inner)&&, Ts...>{rhs.inner.inner},    \
						rhs.inner.tag)} {}

#define VEG_TAGGED_UWUNION_COPY_true(Class)                                    \
	VEG_INLINE                                                                   \
	constexpr Class(Class const& rhs)                                            \
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)))           \
			: inner{_uwunion::make<                                                  \
						decltype(inner),                                                   \
						sizeof...(Ts),                                                     \
						VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>))>(                    \
						UwunionGetter<decltype(inner.inner) const&, Ts...>{                \
								rhs.inner.inner},                                              \
						rhs.inner.tag)} {}

#else

#define VEG_TAGGED_UWUNION_MOVE_true(Class)                                    \
	VEG_INLINE                                                                   \
	Class(Class&& rhs) /* NOLINT */                                              \
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)))            \
			: inner{} {                                                              \
		using Fn = UwunionGetter<decltype(inner.inner)&&, Ts...>;                  \
		inner.tag = rhs.inner.tag;                                                 \
		_detail::visit<                                                            \
				void,                                                                  \
				VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)),                          \
				sizeof...(Ts)>(                                                        \
				this->inner.tag,                                                       \
				CtorFn<decltype(inner.inner), Fn>{inner.inner, Fn{rhs.inner.inner}});  \
	}

#define VEG_TAGGED_UWUNION_COPY_true(Class)                                    \
	VEG_INLINE                                                                   \
	Class(Class const& rhs)                                                      \
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)))           \
			: inner{} {                                                              \
		using Fn = UwunionGetter<decltype(inner.inner) const&, Ts...>;             \
		inner.tag = rhs.inner.tag;                                                 \
		_detail::visit<                                                            \
				void,                                                                  \
				VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)),                         \
				sizeof...(Ts)>(                                                        \
				this->inner.tag,                                                       \
				CtorFn<decltype(inner.inner), Fn>{inner.inner, Fn{rhs.inner.inner}});  \
	}

#endif

#define VEG_TAGGED_UWUNION_DEF(NeedsDtor)                                      \
	template <typename... Ts>                                                    \
	struct NonTrivialUwunionDtor</* NOLINT */                                    \
	                             NeedsDtor,                                      \
	                             Ts...> {                                        \
		using TagType = meta::if_t<sizeof...(Ts) < 256U, u8, usize>;               \
		template <usize I>                                                         \
		using Ith = ith<I, Ts...>;                                                 \
                                                                               \
		union {                                                                    \
			Empty _;                                                                 \
			RawUwunion<Ts...> inner;                                                 \
		};                                                                         \
		TagType tag;                                                               \
                                                                               \
		VEG_INLINE VEG_CPP20(constexpr) NonTrivialUwunionDtor /* NOLINT */         \
				() VEG_NOEXCEPT : _{} {}                                               \
		template <usize I, typename Fn>                                            \
		VEG_INLINE constexpr NonTrivialUwunionDtor(                                \
				EmplaceTag /*etag*/, UTag<I> itag, Fn fn, usize i = I)                 \
				VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{}))                              \
				: inner{itag, itag, VEG_FWD(fn)}, tag(TagType(i)) {}                   \
                                                                               \
		__VEG_PP_CAT(VEG_TAGGED_UWUNION_DTOR_, NeedsDtor)                          \
                                                                               \
		template <usize I>                                                         \
				VEG_INLINE constexpr auto get_ref() const                              \
				& VEG_NOEXCEPT -> ith<I, Ts...> const& {                               \
			return (void)meta::unreachable_if(tag != I),                             \
			       UwunionGetImpl<I>::get(inner).inner;                              \
		}                                                                          \
	}

VEG_TAGGED_UWUNION_DEF(true);
VEG_TAGGED_UWUNION_DEF(false);

#undef VEG_TAGGED_UWUNION_DEF

template <bool NeedsMove, bool NeedsCopy, typename... Ts>
struct NonTrivialUwunionCopyMove;

#define VEG_TAGGED_UWUNION_DEF(NeedsMove, NeedsCopy)                           \
	template <typename... Ts>                                                    \
	struct NonTrivialUwunionCopyMove</* NOLINT */                                \
	                                 NeedsMove,                                  \
	                                 NeedsCopy,                                  \
	                                 Ts...> {                                    \
		template <usize I>                                                         \
		using Ith = ith<I, Ts...>;                                                 \
                                                                               \
		NonTrivialUwunionDtor<                                                     \
				!VEG_ALL_OF(VEG_CONCEPT(trivially_destructible<Ts>)),                  \
				Ts...>                                                                 \
				inner;                                                                 \
                                                                               \
		VEG_INLINE VEG_CPP20(constexpr) NonTrivialUwunionCopyMove() = default;     \
		template <usize I, typename Fn>                                            \
		VEG_INLINE constexpr NonTrivialUwunionCopyMove(                            \
				EmplaceTag etag, UTag<I> itag, Fn fn, usize i = I)                     \
				VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{}))                              \
				: inner{etag, itag, VEG_FWD(fn), i} {}                                 \
                                                                               \
		auto operator=(NonTrivialUwunionCopyMove&&)                                \
				-> NonTrivialUwunionCopyMove& = default;                               \
		auto operator=(NonTrivialUwunionCopyMove const&)                           \
				-> NonTrivialUwunionCopyMove& = default;                               \
		__VEG_PP_CAT(VEG_TAGGED_UWUNION_MOVE_, NeedsMove)                          \
		(NonTrivialUwunionCopyMove) __VEG_PP_CAT(                                  \
				VEG_TAGGED_UWUNION_COPY_, NeedsCopy)(NonTrivialUwunionCopyMove)        \
	}

VEG_TAGGED_UWUNION_DEF(true, true);
VEG_TAGGED_UWUNION_DEF(false, true);
VEG_TAGGED_UWUNION_DEF(true, false);
VEG_TAGGED_UWUNION_DEF(false, false);

template <typename T>
struct IdxMoveFn {
	T&& value;
	VEG_INLINE constexpr auto operator()(void* /*unused*/)
			const&& VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T {
		return T(VEG_FWD(value));
	}
};

template <typename Base, bool NeedsCopy, typename... Ts>
struct NonTrivialCopyAssign {
	NonTrivialCopyAssign() = default;
	~NonTrivialCopyAssign() = default;
	NonTrivialCopyAssign(NonTrivialCopyAssign const& _rhs) = default;
	NonTrivialCopyAssign(NonTrivialCopyAssign&& _rhs) = default;
	auto operator=(NonTrivialCopyAssign&& _rhs)
			-> NonTrivialCopyAssign& = default;
	VEG_INLINE
	VEG_CPP14(constexpr)
	auto operator=(NonTrivialCopyAssign const& _rhs) /* NOLINT */ VEG_NOEXCEPT_IF(
			VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)) &&
			VEG_ALL_OF(VEG_CONCEPT(nothrow_copy_assignable<Wrapper<Ts>>)))
			-> NonTrivialCopyAssign& {
		using Inner = decltype(static_cast<Base*>(this)->inner.inner);
		using Fn = UwunionGetter<Inner const&, Ts...>;
		auto& self = static_cast<Base&>(*this);
		auto& rhs = static_cast<Base const&>(_rhs);
		using NestedInner = decltype(self.inner.inner);

		if (self.inner.tag == rhs.inner.tag) {
			_detail::visit<
					void,
					VEG_ALL_OF(VEG_CONCEPT(nothrow_copy_assignable<Wrapper<Ts>>)),
					sizeof...(Ts)>(
					self.inner.tag,
					AssignFn<NestedInner, Fn, Ts...>{
							self.inner.inner,
							Fn{rhs.inner.inner},
					});
		} else {
			Base local{rhs};
			self.inner.destroy();
			self.inner.tag = rhs.inner.tag;
			self.construct(UwunionGetter<Inner&&, Ts...>{local.inner.inner});
		}
		return *this;
	}
};
template <typename Base, bool NeedsMove, typename... Ts>
struct NonTrivialMoveAssign {
	NonTrivialMoveAssign() = default;
	~NonTrivialMoveAssign() = default;
	NonTrivialMoveAssign(NonTrivialMoveAssign const& _rhs) = default;
	NonTrivialMoveAssign(NonTrivialMoveAssign&& _rhs) = default;
	auto operator=(NonTrivialMoveAssign const& _rhs)
			-> NonTrivialMoveAssign& = default;
	VEG_INLINE
	VEG_CPP14(constexpr)
	auto operator=(NonTrivialMoveAssign&& _rhs) VEG_NOEXCEPT_IF(
			VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)) &&
			VEG_ALL_OF(VEG_CONCEPT(nothrow_move_assignable<Wrapper<Ts>>)))
			-> NonTrivialMoveAssign& {
		using Inner = decltype(static_cast<Base*>(this)->inner.inner);
		using Fn = UwunionGetter<Inner&&, Ts...>;
		auto& self = static_cast<Base&>(*this);
		auto& rhs = static_cast<Base&>(_rhs);
		using NestedInner = decltype(self.inner.inner);

		if (self.inner.tag == rhs.inner.tag) {
			_detail::visit<
					void,
					VEG_ALL_OF(VEG_CONCEPT(nothrow_move_assignable<Wrapper<Ts>>)),
					sizeof...(Ts)>(
					self.inner.tag,
					AssignFn<NestedInner, Fn, Ts...>{
							self.inner.inner,
							Fn{rhs.inner.inner},
					});
		} else {
			self.inner.destroy();
			self.inner.tag = rhs.inner.tag;
			self.construct(UwunionGetter<Inner&&, Ts...>{rhs.inner.inner});
		}
		return *this;
	}
};
template <typename Base, typename... Ts>
struct NonTrivialMoveAssign<Base, false, Ts...> {};
template <typename Base, typename... Ts>
struct NonTrivialCopyAssign<Base, false, Ts...> {};

template <bool NeedsMoveAssign, bool NeedsCopyAssign, typename T>
struct NoOpCopyMove;

template <typename T>
struct NoOpCopyMove<false, false, T> : T {
	using T::T;
};
template <typename T>
struct NoOpCopyMove<false, true, T> : T { /* NOLINT */
	using T::T;
	NoOpCopyMove(NoOpCopyMove const&) = default;
	NoOpCopyMove(NoOpCopyMove&&) = default;
	auto operator=(NoOpCopyMove&& /*unused*/) VEG_NOEXCEPT
			-> NoOpCopyMove& = default;
	auto operator= /* NOLINT */(NoOpCopyMove const& /*unused*/) VEG_NOEXCEPT
			-> NoOpCopyMove& {
		return *this;
	};
};
template <typename T>
struct NoOpCopyMove<true, false, T> : T { /* NOLINT */
	using T::T;
	NoOpCopyMove(NoOpCopyMove const&) = default;
	NoOpCopyMove(NoOpCopyMove&&) = default;
	auto operator=(NoOpCopyMove&& /*unused*/) VEG_NOEXCEPT -> NoOpCopyMove& {
		return *this;
	};
	auto operator=(NoOpCopyMove const& /*unused*/) VEG_NOEXCEPT
			-> NoOpCopyMove& = default;
};
template <typename T>
struct NoOpCopyMove<true, true, T> : T { /* NOLINT */
	using T::T;
	NoOpCopyMove(NoOpCopyMove const&) = default;
	NoOpCopyMove(NoOpCopyMove&&) = default;
	auto operator=(NoOpCopyMove&& /*unused*/) VEG_NOEXCEPT -> NoOpCopyMove& {
		return *this;
	};
	auto operator= /* NOLINT */(NoOpCopyMove const& /*unused*/) VEG_NOEXCEPT
			-> NoOpCopyMove& {
		return *this;
	};
};

#define VEG_NEEDS_MOVE                                                         \
	(VEG_ALL_OF(VEG_CONCEPT(movable<Ts>)) &&                                     \
	 !VEG_ALL_OF(VEG_CONCEPT(trivially_move_constructible<Ts>)))

#define VEG_NEEDS_COPY                                                         \
	(VEG_ALL_OF(VEG_CONCEPT(copyable<Ts>)) &&                                    \
	 !VEG_ALL_OF(VEG_CONCEPT(trivially_copy_constructible<Ts>)))

#define VEG_NEEDS_MOVE_ASSIGN                                                  \
	(!VEG_ALL_OF(VEG_CONCEPT(trivially_move_assignable<Wrapper<Ts>>)) &&         \
	 VEG_ALL_OF(                                                                 \
			 (VEG_CONCEPT(movable<Ts>) &&                                            \
	      VEG_CONCEPT(move_assignable<Wrapper<Ts>>))))

#define VEG_NEEDS_COPY_ASSIGN                                                  \
	(!VEG_ALL_OF(VEG_CONCEPT(trivially_copy_assignable<Wrapper<Ts>>)) &&         \
	 VEG_ALL_OF(                                                                 \
			 (VEG_CONCEPT(copyable<Ts>) &&                                           \
	      VEG_CONCEPT(copy_assignable<Wrapper<Ts>>))))

template <usize... Is, typename... Ts>
struct NonTrivialUwunionImpl<false, meta::index_sequence<Is...>, Ts...>
		: NoOpCopyMove<
					VEG_NEEDS_MOVE_ASSIGN,
					VEG_NEEDS_COPY_ASSIGN,
					NonTrivialUwunionCopyMove<VEG_NEEDS_MOVE, VEG_NEEDS_COPY, Ts...>>,
			NonTrivialMoveAssign<
					NonTrivialUwunionImpl<false, meta::index_sequence<Is...>, Ts...>,
					VEG_NEEDS_MOVE_ASSIGN,
					Ts...>,
			NonTrivialCopyAssign<
					NonTrivialUwunionImpl<false, meta::index_sequence<Is...>, Ts...>,
					VEG_NEEDS_COPY_ASSIGN,
					Ts...> {
	template <usize I>
	using Ith = ith<I, Ts...>;

	VEG_INLINE constexpr auto get_union_ref() const VEG_NOEXCEPT
			-> RawUwunion<Ts...> const& {
		return this->inner.inner;
	}

	using Self = NonTrivialUwunionImpl;
	using Base = NoOpCopyMove<
			VEG_NEEDS_MOVE_ASSIGN,
			VEG_NEEDS_COPY_ASSIGN,
			NonTrivialUwunionCopyMove<VEG_NEEDS_MOVE, VEG_NEEDS_COPY, Ts...>>;

	// precondition: fn(UTag<inner.tag>{}) must be valid
	template <typename Fn>
	VEG_INLINE VEG_CPP20(constexpr) void construct(Fn fn)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_IS_NOEXCEPT(VEG_FWD(fn)(UTag<Is>{})))) {
		_detail::visit<
				void,
				VEG_ALL_OF(VEG_IS_NOEXCEPT(VEG_FWD(fn)(UTag<Is>{}))),
				sizeof...(Ts)>(
				this->inner.tag,
				CtorFn<RawUwunion<Ts...>, Fn>{this->inner.inner, VEG_FWD(fn)});
	}

	using Base::Base;

	template <usize I, typename Fn>
	VEG_INLINE VEG_CPP20(constexpr) void _emplace(Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{})) {
		using T = Ith<I>;
		T local{VEG_FWD(fn)(UTag<I>{})};
		{
			// this block never throws
			this->inner.destroy();
			mem::construct_at(
					mem::addressof(this->inner.inner),
					UTag<I>{},
					UTag<I>{},
					IdxMoveFn<T>{VEG_FWD(local)});
		}
		this->inner.tag = I;
	}

	template <usize I>
			VEG_INLINE constexpr auto get_ref() const
			& VEG_NOEXCEPT -> ith<I, Ts...> const& {
		return (void)meta::unreachable_if(this->inner.tag != I),
		       UwunionGetImpl<I>::get(this->inner.inner).inner;
	}
	VEG_NODISCARD VEG_INLINE constexpr auto index() const -> isize {
		return isize(this->inner.tag);
	}
};

template <typename... Ts>
struct DoubleStorageDtor { /* NOLINT */
	using TagType = meta::if_t<sizeof...(Ts) < 128U, u8, usize>;
	template <usize I>
	using Ith = ith<I, Ts...>;

	union {
		Empty _0;
		RawUwunion<Ts...> inner0;
	};
	union {
		Empty _1;
		RawUwunion<Ts...> inner1;
	};
	TagType tag_with_bit;

	VEG_INLINE VEG_CPP20(constexpr) DoubleStorageDtor() VEG_NOEXCEPT : _0{} {}
	template <usize I, typename Fn>
	VEG_INLINE constexpr DoubleStorageDtor(
			EmplaceTag /*etag*/, UTag<I> itag, Fn fn, usize i = I)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{}))
			: inner0{itag, itag, VEG_FWD(fn)}, tag_with_bit(TagType(2U * i)) {}

	VEG_CPP20(constexpr)
	void destroy()
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>))) {
		_detail::visit14<
				void,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>)),
				sizeof...(Ts)>(
				tag_with_bit / 2U,
				DropFn<
						VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>)),
						decltype(inner0),
						Ts...>{tag_with_bit % 2U == 0 ? inner0 : inner1});
	}
	VEG_CPP20(constexpr)
	void destroy_inactive(usize tag)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>))) {
		_detail::visit14<
				void,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>)),
				sizeof...(Ts)>(
				tag,
				DropFn<
						VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>)),
						decltype(inner0),
						Ts...>{tag_with_bit % 2U == 0 ? inner1 : inner0});
	}

	VEG_INLINE VEG_CPP20(constexpr) ~DoubleStorageDtor()
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>))) {
		this->destroy();
	}
};

template <typename... Ts>
struct DoubleStorageCopyMove { /* NOLINT */
	using TagType = meta::if_t<sizeof...(Ts) < 128U, u8, usize>;
	template <usize I>
	using Ith = ith<I, Ts...>;

	DoubleStorageDtor<Ts...> inner;

	DoubleStorageCopyMove() = default;
	template <usize I, typename Fn>
	VEG_INLINE constexpr DoubleStorageCopyMove(
			EmplaceTag etag, UTag<I> itag, Fn fn, usize i = I)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{}))
			: inner{etag, itag, VEG_FWD(fn), i} {}

#if __cplusplus >= 201703L
	VEG_INLINE constexpr DoubleStorageCopyMove(DoubleStorageCopyMove&& rhs)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)))
			: inner{_uwunion::make<
						decltype(inner),
						sizeof...(Ts),
						VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>))>(
						UwunionGetter<decltype(inner.inner0)&&, Ts...>{
								(rhs.inner.tag_with_bit % 2U == 0) ? rhs.inner.inner0
																									 : rhs.inner.inner1},
						rhs.inner.tag_with_bit / 2U)} {}

	VEG_INLINE constexpr DoubleStorageCopyMove(DoubleStorageCopyMove const& rhs)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)))
			: inner{_uwunion::make<
						decltype(inner),
						sizeof...(Ts),
						VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>))>(
						UwunionGetter<decltype(inner) const&, Ts...>{
								(rhs.inner.tag_with_bit % 2U == 0) ? rhs.inner.inner0
																									 : rhs.inner.inner1,
						},
						rhs.inner.tag_with_bit / 2U)} {}
#else
	VEG_INLINE
	VEG_CPP20(constexpr)
	DoubleStorageCopyMove(DoubleStorageCopyMove&& rhs)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)))
			: inner{} {
		using Fn = UwunionGetter<decltype(inner.inner0)&&, Ts...>;
		inner.tag_with_bit = rhs.inner.tag_with_bit / 2U * 2U;
		_detail::visit<
				void,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)),
				sizeof...(Ts)>(
				inner.tag_with_bit / 2U,
				CtorFn<decltype(inner.inner0), Fn>{
						inner.inner0,
						Fn{(rhs.inner.tag_with_bit % 2U == 0) ? rhs.inner.inner0
		                                              : rhs.inner.inner1}});
	}

	VEG_INLINE
	VEG_CPP20(constexpr)
	DoubleStorageCopyMove(DoubleStorageCopyMove const& rhs)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)))
			: inner{} {
		using Fn = UwunionGetter<decltype(inner.inner0) const&, Ts...>;
		inner.tag_with_bit = rhs.inner.tag_with_bit / 2U * 2U;
		_detail::visit<
				void,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_copyable<Ts>)),
				sizeof...(Ts)>(
				inner.tag_with_bit / 2U,
				CtorFn<decltype(inner.inner0), Fn>{
						inner.inner0,
						Fn{(rhs.inner.tag_with_bit % 2U == 0) ? rhs.inner.inner0
		                                              : rhs.inner.inner1}});
	}
#endif

	template <usize I, typename Fn>
	VEG_INLINE VEG_CPP20(constexpr) void _emplace(Fn fn)
			VEG_NOEXCEPT_LIKE(VEG_FWD(fn)(UTag<I>{})) {
		auto& self_inner_inactive =
				(inner.tag_with_bit % 2U == 0) ? inner.inner1 : inner.inner0;

		// if this throws, object is not yet constructed, and lifetime does not
		// begin. we are still in a valid state
		mem::construct_at(
				mem::addressof(self_inner_inactive), UTag<I>{}, UTag<I>{}, VEG_FWD(fn));

		// invalid state here: two active members
		usize new_tag = I * 2U;
		usize old_tag = inner.tag_with_bit / 2U;
		usize old_low_bit = usize(inner.tag_with_bit % 2U == 1);
		usize new_low_bit = 1U - old_low_bit;
		inner.tag_with_bit = new_tag | new_low_bit;

		// dtor ends the lifetime of the object regardless of whether it throws
		// the call puts us in a valid state regardless of success (i hope)
		inner.destroy_inactive(old_tag);

		// defintely valid again, yay!
	}

	template <bool NoExcept, typename Fn>
	VEG_INLINE VEG_CPP14(constexpr) void assign(usize fn_tag, Fn fn)
			VEG_NOEXCEPT_IF(NoExcept) {
		auto& self_inner_inactive =
				(inner.tag_with_bit % 2U == 0) ? inner.inner1 : inner.inner0;

		auto& self_inner =
				(inner.tag_with_bit % 2U == 0) ? inner.inner0 : inner.inner1;

		usize self_tag = inner.tag_with_bit / 2U;

		if (self_tag == fn_tag) {
			// see above for justification

			_detail::visit<
					void,
					VEG_ALL_OF(VEG_CONCEPT(nothrow_move_assignable<Wrapper<Ts>>)),
					sizeof...(Ts)>(
					self_tag,
					AssignFn<decltype(inner.inner0), Fn, Ts...>{self_inner, VEG_FWD(fn)});
		} else {
			_detail::visit<
					void,
					VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>)),
					sizeof...(Ts)>(
					fn_tag,
					CtorFn<decltype(inner.inner0), Fn>{self_inner_inactive, VEG_FWD(fn)});

			usize new_tag = fn_tag * 2U;
			usize old_tag = self_tag;
			usize old_low_bit = usize(inner.tag_with_bit % 2U == 1);
			usize new_low_bit = 1U - old_low_bit;
			inner.tag_with_bit = TagType(new_tag | new_low_bit);

			inner.destroy_inactive(old_tag);
		}
	}

	VEG_INLINE
	VEG_CPP14(constexpr)
	auto operator=(DoubleStorageCopyMove&& rhs) VEG_NOEXCEPT_IF(VEG_ALL_OF(
			(VEG_CONCEPT(nothrow_move_assignable<Wrapper<Ts>>) &&
	     VEG_CONCEPT(nothrow_movable<Ts>)))) -> DoubleStorageCopyMove& {
		this->template assign<VEG_ALL_OF(
				(VEG_CONCEPT(nothrow_move_assignable<Wrapper<Ts>>) &&
		     VEG_CONCEPT(nothrow_movable<Ts>)))>(
				rhs.inner.tag_with_bit / 2U,
				UwunionGetter<decltype(inner.inner0)&&, Ts...>{
						(rhs.inner.tag_with_bit % 2U == 0) ? rhs.inner.inner0
																							 : rhs.inner.inner1});
		return *this;
	}
	VEG_INLINE
	VEG_CPP14(constexpr)
	auto operator=(DoubleStorageCopyMove const& rhs) VEG_NOEXCEPT_IF(VEG_ALL_OF(
			(VEG_CONCEPT(nothrow_copy_assignable<Wrapper<Ts>>) &&
	     VEG_CONCEPT(nothrow_copyable<Ts>)))) -> DoubleStorageCopyMove& {
		this->template assign<VEG_ALL_OF(
				(VEG_CONCEPT(nothrow_copy_assignable<Wrapper<Ts>>) &&
		     VEG_CONCEPT(nothrow_copyable<Ts>)))>(
				rhs.inner.tag_with_bit / 2U,
				UwunionGetter<decltype(inner.inner0) const&, Ts...>{
						(rhs.inner.tag_with_bit % 2U == 0) ? rhs.inner.inner0
																							 : rhs.inner.inner1});
		return *this;
	}

	VEG_NODISCARD VEG_INLINE constexpr auto index() const VEG_NOEXCEPT -> isize {
		return isize(inner.tag_with_bit / 2U);
	}
	VEG_INLINE constexpr auto get_union_ref() const VEG_NOEXCEPT
			-> RawUwunion<Ts...> const& {
		return (inner.tag_with_bit % 2U == 0) ? inner.inner0 : inner.inner1;
	}
	template <usize I>
			VEG_INLINE constexpr auto get_ref() const
			& VEG_NOEXCEPT -> ith<I, Ts...> const& {
		return UwunionGetImpl<I>::get(get_union_ref()).inner;
	}
};

template <usize... Is, typename... Ts>
struct NonTrivialUwunionImpl<true, meta::index_sequence<Is...>, Ts...>
		: DoubleStorageCopyMove<Ts...>,
			meta::
					if_t<VEG_ALL_OF(VEG_CONCEPT(movable<Ts>)), EmptyI<1312>, NoMoveCtor>,
			meta::
					if_t<VEG_ALL_OF(VEG_CONCEPT(copyable<Ts>)), EmptyI<1313>, NoCopyCtor>,
			meta::if_t<
					VEG_ALL_OF(
							(VEG_CONCEPT(movable<Ts>) &&
               VEG_CONCEPT(move_assignable<Wrapper<Ts>>))),
					EmptyI<1314>,
					NoMoveAssign>,
			meta::if_t<
					VEG_ALL_OF(
							(VEG_CONCEPT(copyable<Ts>) &&
               VEG_CONCEPT(copy_assignable<Wrapper<Ts>>))),
					EmptyI<1315>,
					NoCopyAssign> {
	using Base = DoubleStorageCopyMove<Ts...>;
	using Base::Base;
};

template <typename... Ts>
struct NonTrivialUwunionImplSelector {
	using type = NonTrivialUwunionImpl<
			!(VEG_ALL_OF(VEG_CONCEPT(nothrow_movable<Ts>))),
			meta::make_index_sequence<sizeof...(Ts)>,
			Ts...>;
};

template <typename... Ts>
using UwunionImpl = typename meta::if_t<
		VEG_ALL_OF(
				((!VEG_CONCEPT(movable<Wrapper<Ts>>) ||
          VEG_CONCEPT(trivially_move_constructible<Wrapper<Ts>>)) &&
         (!VEG_CONCEPT(copyable<Wrapper<Ts>>) ||
          VEG_CONCEPT(trivially_copy_constructible<Wrapper<Ts>>)) &&
         (!VEG_CONCEPT(move_assignable<Wrapper<Ts>>) ||
          VEG_CONCEPT(trivially_move_assignable<Wrapper<Ts>>)) &&
         (!VEG_CONCEPT(copy_assignable<Wrapper<Ts>>) ||
          VEG_CONCEPT(trivially_copy_assignable<Wrapper<Ts>>)))),
		meta::type_identity<TrivialUwunionImpl<Ts...>>,
		NonTrivialUwunionImplSelector<Ts...>>::type;

template <typename Fn>
struct TaggedFn {
	Fn&& fn;
	VEG_INLINE constexpr auto
	operator()(void* /*unused*/) const&& VEG_DEDUCE_RET(VEG_FWD(fn)());
};

template <bool NoExcept, typename U1, typename U2>
struct Eq {
	U1 const& u1;
	U2 const& u2;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const
			VEG_NOEXCEPT_IF(NoExcept) -> bool {
		return UwunionGetImpl<I>::get(u1).inner == UwunionGetImpl<I>::get(u2).inner;
	}
};
template <typename U>
struct Dbg {
	fmt::BufferMut out;
	U const& u;
	template <usize I>
	VEG_INLINE void operator()(UTag<I> /*unused*/) const VEG_NOEXCEPT_IF(false) {
		return fmt::Debug<decltype(UwunionGetImpl<I>::get(u).inner)>::to_string(
				VEG_FWD(out), ref(UwunionGetImpl<I>::get(u).inner));
	}
};
template <typename... Ts, usize... Is>
void dbg_impl(
		fmt::BufferMut out,
		uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...> const& u)
		VEG_NOEXCEPT_IF(false) {
	_detail::visit14<void, false, sizeof...(Is)>(
			usize(u.index()),
			_detail::_uwunion::Dbg<_detail::_uwunion::RawUwunion<Ts...>>{
					VEG_FWD(out), u.get_union_ref()});
}
} // namespace _uwunion
} // namespace _detail

namespace uwunion {

VEG_TEMPLATE(
		(typename... Ts, typename... Us, usize... Is),
		requires(VEG_ALL_OF(VEG_CONCEPT(eq<Ts, Us>))),
		VEG_INLINE constexpr auto
		operator==,
		(lhs, IndexedUwunion<meta::index_sequence<Is...>, Ts...> const&),
		(rhs, IndexedUwunion<meta::index_sequence<Is...>, Us...> const&))
VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Ts, Us>)))->bool;

template <usize... Is, typename... Ts>
struct IndexedUwunion<meta::index_sequence<Is...>, Ts...>
		: protected _detail::_uwunion::UwunionImpl<Ts...> {

protected:
	using Base = _detail::_uwunion::UwunionImpl<Ts...>;
	using Base::Base;

public:
	VEG_INLINE constexpr auto get_union_ref() const noexcept
			-> _detail::_uwunion::RawUwunion<Ts...> const& {
		return Base::get_union_ref();
	}

	template <isize I>
	VEG_INLINE constexpr IndexedUwunion(Fix<I> /*itag*/, ith<usize{I}, Ts...> arg)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
			: Base{
						_detail::_uwunion::EmplaceTag{},
						_detail::UTag<usize{I}>{},
						_detail::_uwunion::IdxMoveFn<ith<usize{I}, Ts...>>{VEG_FWD(arg)},
						usize{I},
				} {}
	template <typename T, isize I = position_of<T, Ts...>::value>
	VEG_INLINE constexpr IndexedUwunion(
			Tag<T> /*tag*/, meta::type_identity_t<T> arg)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>))
			: Base{
						_detail::_uwunion::EmplaceTag{},
						_detail::UTag<usize{I}>{},
						_detail::_uwunion::IdxMoveFn<T>{VEG_FWD(arg)},
						I,
				} {}

	template <typename T, isize I = position_of<T, Ts...>::value>
	VEG_INLINE constexpr auto holds(Tag<T> /*tag*/) const noexcept -> bool {
		return index() == I;
	}

	VEG_TEMPLATE(
			(isize I, typename Fn),
			requires(VEG_CONCEPT(fn_once<Fn, ith<usize{I}, Ts...>>)),
			VEG_INLINE constexpr IndexedUwunion,
			(/*inplace*/, InPlace<Fix<I>>),
			(fn, Fn))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, ith<usize{I}, Ts...>>))
			: Base{
						_detail::_uwunion::EmplaceTag{},
						_detail::UTag<usize{I}>{},
						_detail::_uwunion::TaggedFn<Fn&&>{VEG_FWD(fn)},
						usize{I},
				} {}
	VEG_TEMPLATE(
			(typename T, typename Fn),
			requires(VEG_CONCEPT(fn_once<Fn, T>)),
			VEG_INLINE constexpr IndexedUwunion,
			(/*inplace*/, InPlace<Tag<T>>),
			(fn, Fn))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, T>))
			: Base{
						_detail::_uwunion::EmplaceTag{},
						_detail::UTag<position_of<T, Ts...>::value>{},
						_detail::_uwunion::TaggedFn<Fn&&>{VEG_FWD(fn)},
						position_of<T, Ts...>::value,
				} {}

	using Base::index;

	VEG_TEMPLATE(
			typename Fn,
			requires(VEG_ALL_OF(
					VEG_CONCEPT(fn_once<Fn, meta::invoke_result_t<Fn, Ts>, Ts>))),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto map,
			(fn, Fn)) && //
			VEG_NOEXCEPT_IF(VEG_ALL_OF(
					VEG_CONCEPT(nothrow_fn_once<Fn, meta::invoke_result_t<Fn, Ts>, Ts>)))
					-> Uwunion<meta::invoke_result_t<Fn, Ts>...> {
		using Target = Uwunion<meta::invoke_result_t<Fn, Ts>...>;
		return _detail::visit14<
				Target,
				VEG_ALL_OF(VEG_CONCEPT(
						nothrow_fn_once<Fn, meta::invoke_result_t<Fn, Ts>, Ts>)),
				sizeof...(Ts)>(
				usize(index()),
				_detail::_uwunion::EmplaceWrapper<
						Target,
						_detail::_uwunion::FnMapWrapper<Fn&&, Ts...>&&>{
						_detail::_uwunion::FnMapWrapper<Fn&&, Ts...>{
								VEG_FWD(fn),
								this->get_union_ref(),
						},
						usize(index()),
				});
	}

	VEG_TEMPLATE(
			typename Fn,
			requires(
					VEG_ALL_OF(VEG_CONCEPT(fn_once<
																 inner_ith<Fn, Is>,
																 meta::invoke_result_t<inner_ith<Fn, Is>, Ts>,
																 Ts>))),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto map_i,
			(fn, Fn)) && //
			VEG_NOEXCEPT_IF(
					VEG_ALL_OF(VEG_CONCEPT(nothrow_fn_once<
																 inner_ith<Fn, Is>,
																 meta::invoke_result_t<inner_ith<Fn, Is>, Ts>,
																 Ts>)))
					-> Uwunion<meta::invoke_result_t<inner_ith<Fn, Is>, Ts>...> {
		using Target = Uwunion<meta::invoke_result_t<inner_ith<Fn, Is>, Ts>...>;
		return _detail::visit14<
				Target,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_fn_once<
															 inner_ith<Fn, Is>,
															 meta::invoke_result_t<inner_ith<Fn, Is>, Ts>,
															 Ts>)),
				sizeof...(Ts)>(
				usize(index()),
				_detail::_uwunion::EmplaceWrapper<
						Target,
						_detail::_uwunion::FnMapIWrapper<Fn&&, Ts...>&&>{
						_detail::_uwunion::FnMapIWrapper<Fn&&, Ts...>{
								VEG_FWD(fn),
								this->get_union_ref(),
						},
						usize(index()),
				});
	}

	VEG_TEMPLATE(
			(typename Fn,
	     typename Ret = meta::coalesce_t<meta::invoke_result_t<Fn, Ts>...>),
			requires(VEG_ALL_OF(VEG_CONCEPT(fn_once<Fn, Ret, Ts>))),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto visit,
			(fn, Fn)) && //
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_fn_once<Fn, Ret, Ts>)))
					-> Ret {
		return _detail::visit14<
				Ret,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_fn_once<Fn, Ret, Ts>)),
				sizeof...(Ts)>(
				usize(index()),
				_detail::_uwunion::FnVisitWrapper<Ret, Fn&&, Ts...>{
						VEG_FWD(fn),
						this->get_union_ref(),
				});
	}

	VEG_TEMPLATE(
			(typename Fn,
	     typename Ret =
	         meta::coalesce_t<meta::invoke_result_t<inner_ith<Fn, Is>, Ts>...>),
			requires(VEG_ALL_OF(VEG_CONCEPT(fn_once<inner_ith<Fn, Is>, Ret, Ts>))),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto visit_i,
			(fn, Fn)) && //
			VEG_NOEXCEPT_IF(VEG_ALL_OF(
					VEG_CONCEPT(nothrow_fn_once<inner_ith<Fn, Is>, Ret, Ts>))) -> Ret {
		return _detail::visit14<
				Ret,
				VEG_ALL_OF(VEG_CONCEPT(nothrow_fn_once<inner_ith<Fn, Is>, Ret, Ts>)),
				sizeof...(Ts)>(
				usize(index()),
				_detail::_uwunion::FnVisitIWrapper<Ret, Fn&&, Ts...>{
						VEG_FWD(fn),
						this->get_union_ref(),
				});
	}

	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_INLINE VEG_CPP20(constexpr) auto emplace,
			(/*itag*/, Fix<I>),
			(arg, ith<usize{I}, Ts...>))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>) &&
			VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>)))
			->RefMut<ith<usize{I}, Ts...>> {
		this->template _emplace<usize{I}>(
				_detail::_uwunion::IdxMoveFn<ith<usize{I}, Ts...>>{VEG_FWD(arg)});

		return mut(const_cast<ith<usize{I}, Ts...>&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner));
	}

	VEG_TEMPLATE(
			(isize I, typename Fn),
			requires(
					usize{I} < sizeof...(Ts) &&
					VEG_CONCEPT(fn_once<Fn, ith<usize{I}, Ts...>>)),
			VEG_INLINE VEG_CPP20(constexpr) auto emplace_with,
			(/*itag*/, Fix<I>),
			(fn, Fn))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn_once<Fn, ith<usize{I}, Ts...>>) &&
			VEG_ALL_OF(VEG_CONCEPT(nothrow_destructible<Ts>)))
			->RefMut<ith<usize{I}, Ts...>> {
		this->template _emplace<usize{I}>(
				_detail::_uwunion::TaggedFn<Fn&&>{VEG_FWD(fn)});
		return mut(const_cast<ith<usize{I}, Ts...>&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner));
	}

	template <isize I>
	void operator[](Fix<I>) const&& = delete;
	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Fix<I>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		VEG_ASSERT(I == index());
		return static_cast<IndexedUwunion&&>(*this).unwrap_unchecked(
				unsafe, Fix<I>{});
	}
	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE constexpr auto
			operator[],
			(/*itag*/, Fix<I>))
	const & VEG_NOEXCEPT->ith<usize{I}, Ts...> const& {
		return (
				VEG_ASSERT(I == index()),
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}
	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Fix<I>))
	&VEG_NOEXCEPT->ith<usize{I}, Ts...>& {
		VEG_ASSERT(I == index());
		return const_cast<ith<usize{I}, Ts...>&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}

	template <typename T>
	void operator[](Tag<T>) const&& = delete;
	VEG_TEMPLATE(
			(typename T, isize I = position_of<T, Ts...>::value),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Tag<T>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		VEG_ASSERT(I == index());
		return static_cast<IndexedUwunion&&>(*this).unwrap_unchecked(
				unsafe, Fix<I>{});
	}
	VEG_TEMPLATE(
			(typename T, isize I = position_of<T, Ts...>::value),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE constexpr auto
			operator[],
			(/*itag*/, Tag<T>))
	const & VEG_NOEXCEPT->ith<usize{I}, Ts...> const& {
		return (
				VEG_ASSERT(I == index()),
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}
	VEG_TEMPLATE(
			(typename T, isize I = position_of<T, Ts...>::value),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Tag<T>))
	&VEG_NOEXCEPT->ith<usize{I}, Ts...>& {
		VEG_ASSERT(I == index());
		return const_cast<ith<usize{I}, Ts...>&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}

	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto unwrap,
			(/*itag*/, Fix<I>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		VEG_ASSERT(I == index());
		return static_cast<IndexedUwunion&&>(*this).unwrap_unchecked(
				unsafe, Fix<I>{});
	}

	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto unwrap_unchecked,
			(/*unsafe_tag*/, Unsafe),
			(/*itag*/, Fix<I>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		meta::unreachable_if(I != index());
		return const_cast<ith<usize{I}, Ts...>&&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}

	void as_ref() const&& = delete;

	VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto as_ref() const & //
			VEG_NOEXCEPT -> Uwunion<Ref<Ts>...> {
		using Raw = _detail::_uwunion::RawUwunion<Ts...>;
		return _detail::visit14<Uwunion<Ref<Ts>...>, true, sizeof...(Ts)>(
				usize(index()),
				_detail::_uwunion::EmplaceWrapper<
						Uwunion<Ref<Ts>...>,
						_detail::_uwunion::UwunionGetterRef<Raw, Ts...>>{
						{const_cast<Raw&>(this->get_union_ref())},
						usize(index()),
				});
	}
	VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto as_mut() //
			VEG_NOEXCEPT -> Uwunion<RefMut<Ts>...> {
		using Raw = _detail::_uwunion::RawUwunion<Ts...>;
		return _detail::visit14<Uwunion<RefMut<Ts>...>, true, sizeof...(Ts)>(
				usize(index()),
				_detail::_uwunion::EmplaceWrapper<
						Uwunion<RefMut<Ts>...>,
						_detail::_uwunion::UwunionGetterMut<Raw&, Ts...>>{
						{const_cast<Raw&>(this->get_union_ref())},
						usize(index()),
				});
	}
	VEG_EXPLICIT_COPY(IndexedUwunion);
};

VEG_TEMPLATE(
		(typename... Ts, typename... Us, usize... Is),
		requires(VEG_ALL_OF(VEG_CONCEPT(eq<Ts, Us>))),
		VEG_INLINE constexpr auto
		operator==,
		(lhs, IndexedUwunion<meta::index_sequence<Is...>, Ts...> const&),
		(rhs, IndexedUwunion<meta::index_sequence<Is...>, Us...> const&))
VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Ts, Us>)))->bool {
	return lhs.index() == rhs.index() && //
	       _detail::visit<
						 bool,
						 VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Ts, Us>)),
						 sizeof...(Is)>(
						 usize(lhs.index()),
						 _detail::_uwunion::Eq<
								 VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Ts, Us>)),
								 _detail::_uwunion::RawUwunion<Ts...>,
								 _detail::_uwunion::RawUwunion<Us...>>{
								 lhs.get_union_ref(),
								 rhs.get_union_ref(),
						 });
}
} // namespace uwunion

template <typename... Ts>
struct Uwunion
		: public uwunion::
					IndexedUwunion<meta::make_index_sequence<sizeof...(Ts)>, Ts...> {
private:
	template <typename... Us>
	friend struct veg::Uwunion;
	template <typename Seq, typename... Us>
	friend struct veg::uwunion::IndexedUwunion;
	template <typename, typename>
	friend struct _detail::_uwunion::EmplaceWrapper;

	using Base =
			uwunion::IndexedUwunion<meta::make_index_sequence<sizeof...(Ts)>, Ts...>;
	using Base::get_union_ref;

public:
	using Base::Base;

	template <isize I>
	void operator[](Fix<I>) const&& = delete;
	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Fix<I>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		VEG_ASSERT(I == this->index());
		return static_cast<Uwunion&&>(*this).unwrap_unchecked(unsafe, Fix<I>{});
	}
	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE constexpr auto
			operator[],
			(/*itag*/, Fix<I>))
	const & VEG_NOEXCEPT->ith<usize{I}, Ts...> const& {
		return (
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}
	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Fix<I>))
	&VEG_NOEXCEPT->ith<usize{I}, Ts...>& {
		VEG_ASSERT(I == this->index());
		return const_cast<ith<usize{I}, Ts...>&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}

	template <typename T>
	void operator[](Tag<T>) const&& = delete;
	VEG_TEMPLATE(
			(typename T, isize I = position_of<T, Ts...>::value),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Tag<T>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		VEG_ASSERT(I == this->index());
		return static_cast<Uwunion&&>(*this).unwrap_unchecked(unsafe, Fix<I>{});
	}
	VEG_TEMPLATE(
			(typename T, isize I = position_of<T, Ts...>::value),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE constexpr auto
			operator[],
			(/*itag*/, Tag<T>))
	const & VEG_NOEXCEPT->ith<usize{I}, Ts...> const& {
		return (
				VEG_ASSERT(I == this->index()),
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}
	VEG_TEMPLATE(
			(typename T, isize I = position_of<T, Ts...>::value),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
			operator[],
			(/*itag*/, Tag<T>))
	&VEG_NOEXCEPT->ith<usize{I}, Ts...>& {
		VEG_ASSERT(I == this->index());
		return const_cast<ith<usize{I}, Ts...>&>(
				_detail::_uwunion::UwunionGetImpl<usize{I}>::get(this->get_union_ref())
						.inner);
	}

	VEG_TEMPLATE(
			(isize I),
			requires(usize{I} < sizeof...(Ts)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto unwrap,
			(/*itag*/, Fix<I>)) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<ith<usize{I}, Ts...>>))
					-> ith<usize{I}, Ts...> {
		VEG_ASSERT(I == this->index());
		return static_cast<Uwunion&&>(*this).unwrap_unchecked(unsafe, Fix<I>{});
	}

	VEG_EXPLICIT_COPY(Uwunion);
	Uwunion() = delete;
};

namespace uwunion {
using veg::Uwunion;
} // namespace uwunion

namespace _detail {
namespace _uwunion {

template <bool NoExcept, typename U1, typename U2>
struct Ord {
	U1 const& u1;
	U2 const& u2;
	template <usize I>
	VEG_INLINE constexpr auto operator()(UTag<I> /*unused*/) const
			VEG_NOEXCEPT_IF(NoExcept) -> cmp::Ordering {
		using T1 = decltype(UwunionGetImpl<I>::get(u1).inner);
		using T2 = decltype(UwunionGetImpl<I>::get(u2).inner);
		return static_cast<cmp::Ordering>( //
				cmp::Ord<T1, T2>::cmp(
						ref(UwunionGetImpl<I>::get(u1).inner),
						ref(UwunionGetImpl<I>::get(u2).inner)));
	}
};

template <typename... Ts, typename... Us, usize... Is>
VEG_NODISCARD VEG_INLINE constexpr auto
ord(uwunion::IndexedUwunion<veg::meta::index_sequence<Is...>, Ts...> const& lhs,
    uwunion::IndexedUwunion<veg::meta::index_sequence<Is...>, Us...> const& rhs)
		VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)))
				-> cmp::Ordering {
	return (lhs.index() != rhs.index())
	           ? cmp::Ord<isize, isize>::cmp(ref(lhs.index()), ref(rhs.index()))
	           : _detail::visit<
									 cmp::Ordering,
									 VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)),
									 sizeof...(Is)>(
									 usize(lhs.index()),
									 Ord<VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)),
	                     RawUwunion<Ts...>,
	                     RawUwunion<Us...>>{
											 lhs.get_union_ref(),
											 rhs.get_union_ref(),
									 });
}

struct OrdIUwunionBase {
	VEG_TEMPLATE(
			(typename... Ts, typename... Us, usize... Is),
			requires(VEG_ALL_OF(VEG_CONCEPT(ord<Ts, Us>))),
			VEG_NODISCARD VEG_INLINE static constexpr auto cmp,
			(lhs, Ref<uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...>>),
			(rhs, Ref<uwunion::IndexedUwunion<meta::index_sequence<Is...>, Us...>>))
	VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)))->cmp::Ordering {
		return _uwunion::ord(lhs.get(), rhs.get());
	}
};
struct OrdUwunionBase {
	VEG_TEMPLATE(
			(typename... Ts, typename... Us, usize... Is),
			requires(VEG_ALL_OF(VEG_CONCEPT(ord<Ts, Us>))),
			VEG_NODISCARD VEG_INLINE static constexpr auto cmp,
			(lhs, Ref<Uwunion<Ts...>>),
			(rhs, Ref<Uwunion<Us...>>))
	VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)))->cmp::Ordering {
		return _uwunion::ord(lhs.get(), rhs.get());
	}
};

struct DbgIUwunionBase {
	template <usize... Is, typename... Ts>
	static void to_string(
			fmt::BufferMut out,
			Ref<uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...>> u)
			VEG_NOEXCEPT_IF(false) {
		_uwunion::dbg_impl(VEG_FWD(out), u.get());
	}
};
struct DbgUwunionBase {
	template <usize... Is, typename... Ts>
	static void to_string(fmt::BufferMut out, Ref<Uwunion<Ts...>> u)
			VEG_NOEXCEPT_IF(false) {
		_uwunion::dbg_impl(VEG_FWD(out), u.get());
	}
};

template <typename File>
struct BinCerealVisitor {
	RefMut<File> f;
	template <typename T>
			void operator()(Ref<T> t) &&
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(aux::cereal::xnothrow_bin_serializable<T, File>)) {
		cereal::BinCereal<T>::serialize_to(VEG_FWD(f), VEG_FWD(t));
	}
};
template <typename File, typename... Ts>
struct BinDeserialVisitor {
	RefMut<File> f;
	template <usize I>
			auto operator()(UTag<I> /*itag*/) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(
					aux::cereal::xnothrow_bin_deserializable_unsafe<ith<I, Ts...>, File>))
					-> Uwunion<Ts...> {
		return {
				inplace[Fix<isize{I}>{}],
				VEG_LAZY_BY_REF(
						cereal::BinCereal<ith<I, Ts...>>::unchecked_deserialize_from(
								unsafe, Tag<ith<I, Ts...>>{}, VEG_FWD(f)))};
	}
};

struct BinCerealUwunionBase {
	VEG_TEMPLATE(
			(typename... Ts, typename File),
			requires(VEG_ALL_OF(VEG_CONCEPT(bin_cereal<Ts, File>))),
			static void serialize_to,
			(f, RefMut<File>),
			(t, Ref<Uwunion<Ts...>>))
	VEG_NOEXCEPT_IF(VEG_ALL_OF(
			VEG_CONCEPT(aux::cereal::xnothrow_bin_serializable<Ts, File>))) {
		using TagType = meta::if_t<sizeof...(Ts) < 256U, u8, usize>;
		cereal::BinCereal<TagType>::serialize_to(
				VEG_FWD(f), ref(TagType(t.get().index())));
		t.get().as_ref().visit(BinCerealVisitor<File>{VEG_FWD(f)});
	}

	VEG_TEMPLATE(
			(typename... Ts, typename File),
			requires(VEG_ALL_OF(VEG_CONCEPT(bin_cereal<Ts, File>))),
			static auto unchecked_deserialize_from,
			(/*unsafe*/, Unsafe),
			(/*tag*/, Tag<Uwunion<Ts...>>),
			(f, RefMut<File>))
	VEG_NOEXCEPT_IF(
			VEG_ALL_OF(VEG_CONCEPT(
					aux::cereal::xnothrow_bin_deserializable_unsafe<Ts, File>)))
			->Uwunion<Ts...> {
		using TagType = meta::if_t<sizeof...(Ts) < 256U, u8, usize>;
		TagType index = cereal::BinCereal<TagType>::unchecked_deserialize_from(
				unsafe, Tag<TagType>{}, mut(f.get()));

		return veg::_detail::visit14<
				Uwunion<Ts...>,
				VEG_ALL_OF(VEG_CONCEPT(
						aux::cereal::xnothrow_bin_deserializable_unsafe<Ts, File>)),
				sizeof...(Ts)>(index, BinDeserialVisitor<File, Ts...>{VEG_FWD(f)});
	}
};
} // namespace _uwunion
} // namespace _detail

template <typename... Ts, typename... Us>
struct cmp::Ord<Uwunion<Ts...>, Uwunion<Us...>>
		: _detail::_uwunion::OrdUwunionBase {};
template <typename... Ts, typename... Us, usize... Is>
struct cmp::Ord<
		uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...>,
		uwunion::IndexedUwunion<meta::index_sequence<Is...>, Us...>>
		: _detail::_uwunion::OrdUwunionBase {};

template <typename... Ts>
struct cereal::BinCereal<Uwunion<Ts...>>
		: _detail::_uwunion::BinCerealUwunionBase {};

template <typename... Ts>
struct fmt::Debug<Uwunion<Ts...>> : _detail::_uwunion::DbgUwunionBase {};
template <usize... Is, typename... Ts>
struct fmt::Debug<uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...>>
		: _detail::_uwunion::DbgIUwunionBase {};

template <usize... Is, typename... Ts>
struct cpo::is_trivially_constructible<
		uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...>>
		: meta::false_type {};
template <typename... Ts>
struct cpo::is_trivially_constructible<uwunion::Uwunion<Ts...>>
		: meta::false_type {};

template <usize... Is, typename... Ts>
struct cpo::is_trivially_relocatable<
		uwunion::IndexedUwunion<meta::index_sequence<Is...>, Ts...>>
		: meta::bool_constant<VEG_ALL_OF(is_trivially_relocatable<Ts>::value)> {};
template <typename... Ts>
struct cpo::is_trivially_relocatable<uwunion::Uwunion<Ts...>>
		: meta::bool_constant<VEG_ALL_OF(is_trivially_relocatable<Ts>::value)> {};
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_UWUNION_HPP_OHC4GK5JS */
