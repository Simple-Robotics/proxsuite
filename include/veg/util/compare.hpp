#ifndef VEG_CMP_HPP_6QBW4XNOS
#define VEG_CMP_HPP_6QBW4XNOS

#include "veg/type_traits/primitives.hpp"
#include "veg/type_traits/core.hpp"
#include "veg/internal/integer_seq.hpp"
#include "veg/ref.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {

namespace cmp {
enum struct Ordering : unsigned char {
	equal = 0,
	less = 1,
	greater = 2,
	unordered = 3,
};

template <typename T, typename U>
struct Ord;

} // namespace cmp

namespace concepts {
VEG_CONCEPT_EXPR(
		(typename T, typename U),
		(T, U),
		legacy_ord,
		VEG_DECLVAL(T const&) < VEG_DECLVAL(U const&),
		VEG_CONCEPT(constructible<bool, ExprType>));

VEG_CONCEPT_EXPR(
		(typename T, typename U),
		(T, U),
		eq,
		VEG_DECLVAL(T const&) == VEG_DECLVAL(U const&),
		VEG_CONCEPT(constructible<bool, ExprType>));

VEG_CONCEPT_EXPR(
		(typename T, typename U),
		(T, U),
		ord,
		(cmp::Ord<T, U>::cmp(VEG_DECLVAL(Ref<T>), VEG_DECLVAL(Ref<U>))),
		VEG_CONCEPT(constructible<cmp::Ordering, ExprType>));
} // namespace concepts

namespace cmp {
namespace ref {
VEG_TEMPLATE(
		(typename T, typename U),
		requires(VEG_CONCEPT(eq<T, U>)),
		VEG_INLINE constexpr auto
		operator==,
		(lhs, Ref<T>),
		(rhs, Ref<U>))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
	return lhs.get() == rhs.get();
}
} // namespace ref
namespace mut {
VEG_TEMPLATE(
		(typename T, typename U),
		requires(VEG_CONCEPT(eq<T, U>)),
		VEG_INLINE constexpr auto
		operator==,
		(lhs, RefMut<T>),
		(rhs, RefMut<U>))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->cmp::Ordering {
	return static_cast<T const&>(lhs.get()) == static_cast<U const&>(rhs.get());
}
} // namespace mut
} // namespace cmp

namespace _detail {
namespace _cmp {
struct OrdBase {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(legacy_ord<T, U>) && VEG_CONCEPT(eq<T, U>)),
			VEG_INLINE static constexpr auto cmp,
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_legacy_ord<T, U>) && VEG_CONCEPT(nothrow_eq<T, U>))
			->cmp::Ordering {
		return static_cast<bool>(lhs.get() == rhs.get())            //
		           ? cmp::Ordering::equal                           //
		           : static_cast<bool>(lhs.get() < rhs.get())       //
		                 ? cmp::Ordering::less                      //
		                 : static_cast<bool>(lhs.get() > rhs.get()) //
		                       ? cmp::Ordering::greater             //
		                       : cmp::Ordering::unordered;          //
	}
};

struct OrdRefBase {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE static constexpr auto cmp,
			(lhs, Ref<Ref<T>>),
			(rhs, Ref<Ref<U>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->cmp::Ordering {
		return cmp::Ord<T, U>::cmp(lhs.get(), rhs.get());
	}
};

struct OrdRefMutBase {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE static constexpr auto cmp,
			(lhs, Ref<Ref<T>>),
			(rhs, Ref<Ref<U>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->cmp::Ordering {
		return cmp::Ord<T, U>::cmp( //
				lhs.get().as_const(),
				rhs.get().as_const());
	}
};
} // namespace _cmp
} // namespace _detail

namespace cmp {
template <typename T, typename U>
struct Ord : _detail::_cmp::OrdBase {};
template <typename T, typename U>
struct Ord<Ref<T>, Ref<U>> : _detail::_cmp::OrdRefBase {};
template <typename T, typename U>
struct Ord<RefMut<T>, RefMut<U>> : _detail::_cmp::OrdRefMutBase {};

namespace nb {
struct eq {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(eq<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
		return bool(lhs.get() == rhs.get());
	}
};
struct ne {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(eq<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
		return !eq{}(lhs, rhs);
	}
};

struct lt {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->bool {
		return static_cast<Ordering>(Ord<T, U>::cmp(lhs, rhs)) == Ordering::less;
	}
};
struct gt {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->bool {
		return static_cast<Ordering>(Ord<T, U>::cmp(lhs, rhs)) == Ordering::greater;
	}
};
struct leq {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->bool {
		return unsigned(static_cast<Ordering>(Ord<T, U>::cmp(lhs, rhs))) < 2U;
	}
};
struct geq {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->bool {
		return unsigned(static_cast<Ordering>(Ord<T, U>::cmp(lhs, rhs))) % 2U == 0;
	}
};

struct cmp {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_INLINE constexpr auto
			operator(),
			(lhs, Ref<T>),
			(rhs, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->Ordering {
		return static_cast<Ordering>(Ord<T, U>::cmp(lhs, rhs));
	}
};
} // namespace nb
VEG_NIEBLOID(eq);
VEG_NIEBLOID(ne);
VEG_NIEBLOID(lt);
VEG_NIEBLOID(gt);
VEG_NIEBLOID(leq);
VEG_NIEBLOID(geq);
VEG_NIEBLOID(cmp);
} // namespace cmp

namespace _detail {
#if VEG_HAS_FOLD_EXPR == 1 && __cplusplus >= 201402L
template <typename... Ts>
struct CmpImpl {
	template <typename... Us>
	VEG_INLINE static constexpr auto eq(Ts const&... ts, Us const&... us)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Ts, Us>))) -> bool {
		return VEG_ALL_OF((ts == us));
	}

	template <typename... Us>
	VEG_INLINE static constexpr auto cmp(Ts const&... ts, Us const&... us)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)))
					-> cmp::Ordering {

		bool found = false;
		auto c = cmp::Ordering::equal;

		(void)(

				VEG_ALL_OF(
						((void)(c = cmp::cmp(ref(ts), ref(us))),
		         ((c != cmp::Ordering::equal) ? void(found = true) : (void)0),
		         !found))

		);
		return c;
	}
};
#else

template <typename... Ts>
struct CmpImpl;
template <typename T0, typename... Ts>
struct CmpImpl<T0, Ts...> {

	template <typename U0, typename... Us>
	VEG_INLINE static constexpr auto
	eq(T0 const& t0, Ts const&... ts, U0 const& u0, Us const&... us)
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_eq<T0, U0>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Ts, Us>))) -> bool {
		return (t0 == u0) && CmpImpl<Ts...>::eq(ts..., us...);
	}

	template <typename U0, typename... Us>
	VEG_INLINE static constexpr auto
	cmp(T0 const& t0, Ts const&... ts, U0 const& u0, Us const&... us)
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_ord<T0, U0>) &&
					VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>))) -> cmp::Ordering {
		return CmpImpl::cmp2(cmp::cmp(ref(t0), ref(u0)), ts..., us...);
	}

	template <typename... Us>
	VEG_INLINE static constexpr auto
	cmp2(cmp::Ordering res, Ts const&... ts, Us const&... us)
			VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_ord<Ts, Us>)))
					-> cmp::Ordering {
		return (res != cmp::Ordering::equal) ? res
		                                     : CmpImpl<Ts...>::cmp(ts..., us...);
	}
};
template <>
struct CmpImpl<> {
	VEG_INLINE static constexpr auto eq() VEG_NOEXCEPT -> bool { return true; }
	VEG_INLINE static constexpr auto cmp() VEG_NOEXCEPT -> cmp::Ordering {
		return cmp::Ordering::equal;
	}
};

#endif

template <typename T, usize... Is, typename... Bases, typename... Members>
VEG_INLINE constexpr auto reflected_eq_impl(
		T const& lhs,
		T const& rhs,
		_detail::SimpleITuple<
				_detail::_meta::integer_sequence<usize, Is...>,
				Members Bases::*...> member_ptrs)
		VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Members, Members>)))
				-> bool {
	return _detail::CmpImpl<Members...>::eq( //
			lhs.*(static_cast<_detail::SimpleLeaf<Is, Members Bases::*> const&>(
								member_ptrs)
	              .inner)...,
			rhs.*(static_cast<_detail::SimpleLeaf<Is, Members Bases::*> const&>(
								member_ptrs)
	              .inner)...);
}

template <typename T, usize... Is, typename... Bases, typename... Members>
VEG_INLINE constexpr auto reflected_cmp_impl(
		T const& lhs,
		T const& rhs,
		_detail::SimpleITuple<
				_detail::_meta::integer_sequence<usize, Is...>,
				Members Bases::*...> member_ptrs)
		VEG_NOEXCEPT_IF(VEG_ALL_OF(VEG_CONCEPT(nothrow_eq<Members, Members>)))
				-> cmp::Ordering {
	return _detail::CmpImpl<Members...>::cmp( //
			lhs.*(static_cast<_detail::SimpleLeaf<Is, Members Bases::*> const&>(
								member_ptrs)
	              .inner)...,
			rhs.*(static_cast<_detail::SimpleLeaf<Is, Members Bases::*> const&>(
								member_ptrs)
	              .inner)...);
}
} // namespace _detail
namespace cmp {
namespace nb {
struct reflected_eq {
	template <typename T>
	VEG_INLINE constexpr auto operator()(T const& lhs, T const& rhs) const
			VEG_NOEXCEPT_LIKE(_detail::reflected_eq_impl(
					lhs, rhs, _detail::member_extract_access<T>::Type::member_pointers()))
					-> bool {
		return _detail::reflected_eq_impl(
				lhs, rhs, _detail::member_extract_access<T>::Type::member_pointers());
	}
};
struct reflected_cmp {
	template <typename T>
	VEG_INLINE constexpr auto operator()(T const& lhs, T const& rhs) const
			VEG_NOEXCEPT_LIKE(_detail::reflected_cmp_impl(
					lhs, rhs, _detail::member_extract_access<T>::Type::member_pointers()))
					-> veg::cmp::Ordering {
		return _detail::reflected_cmp_impl(
				lhs, rhs, _detail::member_extract_access<T>::Type::member_pointers());
	}
};
} // namespace nb
VEG_NIEBLOID(reflected_eq);
VEG_NIEBLOID(reflected_cmp);

struct ReflectOrd {
	template <typename T>
	VEG_INLINE static constexpr auto cmp(Ref<T> lhs, Ref<T> rhs)
			VEG_NOEXCEPT_LIKE(reflected_cmp(lhs, rhs)) -> cmp::Ordering {
		return reflected_cmp(lhs, rhs);
	}
};
} // namespace cmp
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_CMP_HPP_6QBW4XNOS */
