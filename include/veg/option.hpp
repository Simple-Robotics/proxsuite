#ifndef VEG_OPTION_HPP_8NVLXES2S
#define VEG_OPTION_HPP_8NVLXES2S

#include "veg/util/assert.hpp"
#include "veg/util/unreachable.hpp"
#include "veg/memory/placement.hpp"
#include "veg/memory/address.hpp"
#include "veg/type_traits/constructible.hpp"
#include "veg/type_traits/assignable.hpp"
#include "veg/cereal/bin_cereal.hpp"
#include "veg/util/compare.hpp"
#include "veg/internal/delete_special_members.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {
namespace _detail {
namespace _option {
namespace adl {
struct AdlBase {};
} // namespace adl
} // namespace _option
} // namespace _detail

template <typename T>
struct Option;

namespace option {
namespace nb {
struct some {
	template <typename T>
	VEG_NODISCARD VEG_INLINE constexpr auto operator()(T arg) const
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> Option<T> {
		return {inplace[some{}], _detail::MoveFn<T>{VEG_FWD(arg)}};
	}
};
} // namespace nb
VEG_NIEBLOID(some);
} // namespace option

inline namespace tags {
using Some = veg::option::nb::some;
using veg::option::some;

struct None : _detail::_option::adl::AdlBase {};
VEG_INLINE_VAR(none, None);
} // namespace tags

namespace meta {
template <typename T>
struct is_option : false_type {};
template <typename T>
struct is_option<Option<T>> : true_type {};

template <typename T>
struct option_type : type_identity<_detail::_meta::none> {};
template <typename T>
struct option_type<Option<T>> : type_identity<T> {};

template <typename T>
using option_type_t = typename option_type<T>::type;
} // namespace meta

namespace concepts {
VEG_DEF_CONCEPT(typename T, option, meta::is_option<T>::value);
} // namespace concepts

namespace _detail {
namespace _option {
struct MoveMapFn {
	template <typename T>
	VEG_INLINE constexpr auto operator()(T&& ref) const
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T {
		return VEG_FWD(ref);
	}
};

template <typename T>
struct RetNone {
	VEG_INLINE VEG_CPP14(constexpr) auto operator()() const VEG_NOEXCEPT -> T {
		return none;
	}
};

template <typename Fn>
struct WrapFn {
	using T = meta::invoke_result_t<Fn>;
	Fn fn;
	VEG_CPP14(constexpr)
	auto operator()() && VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, T>))
													 -> Wrapper<T> {
		return Wrapper<T>{VEG_FWD(fn)()};
	}
};
struct DeferMakeNoneIf /* NOLINT */ {
	Empty* none_addr;
	bool success;
	VEG_CPP20(constexpr) ~DeferMakeNoneIf() {
		if (!success) {
			mem::construct_at(none_addr);
		}
	}
};
struct DeferMakeNone /* NOLINT */ {
	Empty* none_addr;
	VEG_CPP20(constexpr) ~DeferMakeNone() { mem::construct_at(none_addr); }
};

template <typename T, bool = VEG_CONCEPT(trivially_destructible<T>)>
struct OptionTrivial {
	union {
		Empty none_val;
		Wrapper<T> some_val;
	};
	bool is_engaged = false;

	constexpr OptionTrivial() VEG_NOEXCEPT : none_val{} {}

	template <typename Fn>
	constexpr OptionTrivial(Some /*some*/, Fn fn)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, T>))
			: some_val{T(VEG_FWD(fn)())}, is_engaged{true} {}

	VEG_CPP20(constexpr)
	void disengage() VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_destructible<T>)) {
		this->is_engaged = false;
		_detail::_option::DeferMakeNone _{&this->none_val};
		mem::destroy_at(mem::addressof(some_val));
	}
};
template <typename T>
struct OptionNonTrivialBase {
	union {
		Empty none_val;
		Wrapper<T> some_val;
	};
	bool is_engaged = false;

	VEG_CPP20(constexpr)
	void disengage() VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_destructible<T>)) {
		this->is_engaged = false;
		_detail::_option::DeferMakeNone _{&this->none_val};
		mem::destroy_at(mem::addressof(some_val));
	}

	VEG_CPP20(constexpr)
	~OptionNonTrivialBase()
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_destructible<T>)) {
		if (is_engaged) {
			mem::destroy_at(mem::addressof(some_val));
		} else {
			mem::destroy_at(&none_val);
		}
	}
	constexpr OptionNonTrivialBase() VEG_NOEXCEPT : none_val{} {}

	template <typename Fn>
	constexpr OptionNonTrivialBase(Some /*some*/, Fn fn)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, T>))
			: some_val{T(VEG_FWD(fn)())}, is_engaged{true} {}

	VEG_CPP20(constexpr)
	OptionNonTrivialBase(OptionNonTrivialBase&& rhs)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) {
		if (rhs.is_engaged) {
			mem::destroy_at(&none_val);
			DeferMakeNoneIf _{&none_val, false};
			mem::construct_at(mem::addressof(some_val), VEG_FWD(rhs.some_val));
			_.success = true;
			is_engaged = true;
		}
	}
	VEG_CPP20(constexpr)
	OptionNonTrivialBase(OptionNonTrivialBase const& rhs)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_copyable<T>)) {
		if (rhs.is_engaged) {
			mem::destroy_at(&none_val);
			DeferMakeNoneIf _{&none_val, false};
			mem::construct_at(mem::addressof(some_val), rhs.some_val);
			_.success = true;
			is_engaged = true;
		}
	}

	VEG_CPP20(constexpr)
	auto operator= /* NOLINT(cert-oop54-cpp) */(OptionNonTrivialBase&& rhs)
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_movable<Wrapper<T>>) &&
					VEG_CONCEPT(nothrow_move_assignable<T>)) -> OptionNonTrivialBase& {
		if (rhs.is_engaged) {
			if (is_engaged) {
				some_val.inner = VEG_FWD(rhs.some_val.inner);
			} else {
				mem::destroy_at(mem::addressof(none_val));
				DeferMakeNoneIf _{&none_val, false};
				mem::construct_at(mem::addressof(some_val), VEG_FWD(rhs.some_val));
				_.success = true;
				is_engaged = true;
			}
		} else {
			if (is_engaged) {
				disengage();
			}
		}
		return *this;
	}
	VEG_CPP20(constexpr)
	auto operator= /* NOLINT(cert-oop54-cpp) */(OptionNonTrivialBase const& rhs)
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_copyable<Wrapper<T>>) &&
					VEG_CONCEPT(nothrow_copy_assignable<T>)) -> OptionNonTrivialBase& {
		if (rhs.is_engaged) {
			if (is_engaged) {
				some_val.inner = rhs.some_val.inner;
			} else {
				mem::destroy_at(mem::addressof(none_val));
				DeferMakeNoneIf _{&none_val, false};
				mem::construct_at(mem::addressof(some_val), rhs.some_val);
				_.success = true;
				is_engaged = true;
			}
		} else {
			if (is_engaged) {
				disengage();
			}
		}
		return *this;
	}
};
template <typename T>
struct OptionNonTrivial
		: meta::if_t<VEG_CONCEPT(copyable<Wrapper<T>>), EmptyI<0>, NoCopy>,
			meta::if_t<VEG_CONCEPT(movable<Wrapper<T>>), EmptyI<1>, NoMove>,
			meta::if_t<VEG_CONCEPT(copy_assignable<T>), EmptyI<2>, NoCopyAssign>,
			meta::if_t<VEG_CONCEPT(move_assignable<T>), EmptyI<3>, NoMoveAssign>,
			OptionNonTrivialBase<T> {
	using OptionNonTrivialBase<T>::OptionNonTrivialBase;
};

} // namespace _option
} // namespace _detail

template <typename T>
struct VEG_NODISCARD Option
		: private meta::if_t<
					((!VEG_CONCEPT(movable<_detail::Wrapper<T>>) ||
            VEG_CONCEPT(trivially_move_constructible<_detail::Wrapper<T>>)) &&
           (!VEG_CONCEPT(copyable<_detail::Wrapper<T>>) ||
            VEG_CONCEPT(trivially_copy_constructible<_detail::Wrapper<T>>)) &&
           (!VEG_CONCEPT(move_assignable<_detail::Wrapper<T>>) ||
            VEG_CONCEPT(trivially_move_assignable<_detail::Wrapper<T>>)) &&
           (!VEG_CONCEPT(copy_assignable<_detail::Wrapper<T>>) ||
            VEG_CONCEPT(trivially_copy_assignable<_detail::Wrapper<T>>))),
					_detail::_option::OptionTrivial<T>,
					_detail::_option::OptionNonTrivial<T>>,

			private _detail::_option::adl::AdlBase {
private:
	using Base = meta::if_t<
			((!VEG_CONCEPT(movable<_detail::Wrapper<T>>) ||
	      VEG_CONCEPT(trivially_move_constructible<_detail::Wrapper<T>>)) &&
	     (!VEG_CONCEPT(copyable<_detail::Wrapper<T>>) ||
	      VEG_CONCEPT(trivially_copy_constructible<_detail::Wrapper<T>>)) &&
	     (!VEG_CONCEPT(move_assignable<_detail::Wrapper<T>>) ||
	      VEG_CONCEPT(trivially_move_assignable<_detail::Wrapper<T>>)) &&
	     (!VEG_CONCEPT(copy_assignable<_detail::Wrapper<T>>) ||
	      VEG_CONCEPT(trivially_copy_assignable<_detail::Wrapper<T>>))),
			_detail::_option::OptionTrivial<T>,
			_detail::_option::OptionNonTrivial<T>>;

public:
	constexpr Option() VEG_NOEXCEPT = default;
	VEG_EXPLICIT_COPY(Option);

	constexpr Option // NOLINT(hicpp-explicit-conversions)
			(None /*tag*/) VEG_NOEXCEPT : Option{} {}

	VEG_INLINE constexpr Option(Some /*tag*/, T value)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>))
			: Base{some, _detail::MoveFn<T>{VEG_FWD(value)}} {}

	VEG_TEMPLATE(
			(typename _, typename Fn),
			requires(VEG_CONCEPT(same<_, Some>) && VEG_CONCEPT(fn_once<Fn, T>)),
			VEG_INLINE constexpr Option,
			(/*tag*/, InPlace<_>),
			(fn, Fn))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, T>))
			: Base{some, VEG_FWD(fn)} {}

	VEG_CPP20(constexpr)
	void reset() VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_destructible<T>)) {
		if (is_some()) {
			this->disengage();
		}
	}

	VEG_CPP20(constexpr)
	auto operator=(None /*arg*/)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_destructible<T>)) -> Option& {
		reset();
		return *this;
	}

	VEG_TEMPLATE(
			typename _ = T,
			requires(VEG_CONCEPT(option<_>)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto flatten,
			(= safe, Safe))
	&&VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>))->T {
		return static_cast<Option<T>&&>(*this).map_or_else(
				_detail::_option::MoveMapFn{}, _detail::_option::RetNone<T>{});
	}

private:
	VEG_CPP14(constexpr) auto _get() const VEG_NOEXCEPT -> T& {
		return const_cast<T&>(this->some_val.inner);
	}

public:
	VEG_CPP20(constexpr)
	auto take() VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> Option<T> {
		if (is_some()) {
			Option<T> val{
					inplace[some],
					_detail::MoveFn<T>{static_cast<T&&>(this->_get())},
			};
			reset();
			return val;
		}
		return none;
	}

	VEG_CPP14(constexpr)
	auto unwrap_unchecked(Unsafe /*tag*/) &&
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T {
		meta::unreachable_if(is_none());
		return static_cast<T&&>(this->_get());
	}

	VEG_CPP14(constexpr)
	auto unwrap() && VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T {
		VEG_INTERNAL_ASSERT_PRECONDITION(is_some());
		return static_cast<T&&>(this->_get());
	}

	VEG_TEMPLATE(
			(typename Fn),
			requires(VEG_CONCEPT(fn_once<Fn, bool, Ref<T>>)),
			VEG_NODISCARD VEG_CPP14(constexpr) auto filter,
			(fn, Fn)) &&

			VEG_NOEXCEPT_IF(
					(VEG_CONCEPT(nothrow_fn_once<Fn, bool, Ref<T>>) &&
	         VEG_CONCEPT(nothrow_movable<T>))) -> Option<T> {
		if (is_some() && VEG_FWD(fn)(ref(this->_get()))) {
			return {
					inplace[some],
					_detail::MoveFn<T>{static_cast<T&&>(this->_get())},
			};
		}
		return none;
	}

	VEG_TEMPLATE(
			(typename U = T),
			requires(VEG_CONCEPT(eq<T, U>)),
			VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto contains,
			(val, Ref<U>))
	const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
		if (is_none()) {
			return false;
		}
		return cmp::eq(ref(_get()), val);
	}

	VEG_TEMPLATE(
			typename Fn,
			requires(VEG_CONCEPT(fn_once<Fn, T>)),
			VEG_CPP20(constexpr) auto emplace_with,
			(fn, Fn)) &
			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_destructible<T>) &&
					VEG_CONCEPT(nothrow_fn_once<Fn, T>)) -> RefMut<T> {
		if (is_some()) {
			this->disengage();
		}
		mem::destroy_at(&this->none_val);
		_detail::_option::DeferMakeNoneIf _{&this->none_val, false};
		mem::construct_with(
				mem::addressof(this->some_val),
				_detail::_option::WrapFn<Fn>{VEG_FWD(fn)});
		_.success = true;
		this->is_engaged = true;
		return mut(this->_get());
	}

	VEG_CPP14(constexpr)
	auto emplace(T value) VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T& {
		return emplace_with(_detail::MoveFn<T>{VEG_FWD(value)});
	}

	VEG_TEMPLATE(
			(typename Fn, typename Ret = meta::invoke_result_t<Fn, T>),
			requires(VEG_CONCEPT(fn_once<Fn, Ret, T>)),
			VEG_CPP14(constexpr) auto map,
			(fn, Fn)) &&

			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, Ret, T>)) -> Option<Ret> {
		if (is_some()) {
			return Option<Ret>{
					inplace[some],
					_detail::WithArg<Fn&&, T&&>{
							VEG_FWD(fn), static_cast<T&&>(this->_get())},
			};
		}
		return none;
	}

	VEG_TEMPLATE(
			(typename Fn, typename Ret = meta::invoke_result_t<Fn, T>),
			requires(
					VEG_CONCEPT(fn_once<Fn, Ret, T>) && //
					VEG_CONCEPT(option<Ret>)),
			VEG_NODISCARD VEG_CPP14(constexpr) auto and_then,
			(fn, Fn)) &&

			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, Ret, T>)) -> Ret {
		if (is_some()) {
			return VEG_FWD(fn)(static_cast<T&&>(this->_get()));
		}
		return none;
	}

	VEG_TEMPLATE(
			(typename Fn, typename D, typename Ret = meta::invoke_result_t<Fn, T>),
			requires(
					VEG_CONCEPT(fn_once<Fn, Ret, T>) && //
					VEG_CONCEPT(fn_once<D, Ret>)),
			VEG_CPP14(constexpr) auto map_or_else,
			(fn, Fn),
			(d, D)) &&

			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_fn_once<Fn, Ret, T>) &&
					VEG_CONCEPT(nothrow_fn_once<D, Ret>)) -> Ret {
		if (is_some()) {
			return VEG_FWD(fn)(static_cast<T&&>(this->_get()));
		}
		return VEG_FWD(d)();
	}

	VEG_TEMPLATE(
			(typename Fn, typename Ret = meta::invoke_result_t<Fn, T>),
			requires(VEG_CONCEPT(fn_once<Fn, Ret, T>)),
			VEG_CPP14(constexpr) auto map_or,
			(fn, Fn),
			(d, DoNotDeduce<Ret>)) &&

			VEG_NOEXCEPT_IF(
					VEG_CONCEPT(nothrow_fn_once<Fn, Ret, T>) &&
					VEG_CONCEPT(nothrow_movable<Ret>)) -> Ret {
		if (is_none()) {
			return VEG_FWD(d);
		}
		return VEG_FWD(fn)(static_cast<T&&>(this->_get()));
	}

	VEG_TEMPLATE(
			(typename Fn),
			requires(VEG_CONCEPT(fn_once<Fn, Option<T>>)),
			VEG_NODISCARD VEG_CPP14(constexpr) auto or_else,
			(fn, Fn)) &&

			VEG_NOEXCEPT_IF(
					(VEG_CONCEPT(nothrow_fn_once<Fn, Option<T>>) &&
	         VEG_CONCEPT(nothrow_movable<T>))) -> Option<T> {
		if (is_some()) {
			return {
					inplace[some],
					_detail::MoveFn<T>{static_cast<T&&>(this->_get())},
			};
		}
		return VEG_FWD(fn)();
	}

	VEG_NODISCARD VEG_INLINE constexpr auto is_some() const VEG_NOEXCEPT -> bool {
		return this->is_engaged;
	}
	VEG_NODISCARD VEG_INLINE constexpr auto is_none() const VEG_NOEXCEPT -> bool {
		return !is_some();
	}

	VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto as_mut() VEG_NOEXCEPT
			-> Option<RefMut<T>> {
		if (is_some()) {
			return {some, veg::mut(this->_get())};
		}
		return {};
	}
	VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto as_ref() const
			& VEG_NOEXCEPT -> Option<Ref<T>> {
		if (is_some()) {
			return {some, ref(this->_get())};
		}
		return {};
	}
};
namespace _detail {
namespace _option {
namespace adl {
VEG_TEMPLATE(
		(typename T, typename U),
		requires(VEG_CONCEPT(eq<T, U>)),
		VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto
		operator==,
		(lhs, Option<T> const&),
		(rhs, Option<U> const&))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
	if (lhs.is_some() && rhs.is_some()) {
		return cmp::eq(
				lhs.as_ref().unwrap_unchecked(unsafe),
				rhs.as_ref().unwrap_unchecked(unsafe));
	}
	return (lhs.is_some() == rhs.is_some());
}
template <typename T>
VEG_NODISCARD VEG_INLINE constexpr auto
operator==(Option<T> const& lhs, None /*rhs*/) VEG_NOEXCEPT -> bool {
	return lhs.is_none();
}
template <typename U>
VEG_NODISCARD VEG_INLINE constexpr auto
operator==(None /*lhs*/, Option<U> const& rhs) VEG_NOEXCEPT -> bool {
	return rhs.is_none();
}
constexpr auto operator==(None /*lhs*/, None /*rhs*/) VEG_NOEXCEPT -> bool {
	return true;
}
} // namespace adl
} // namespace _option
} // namespace _detail

namespace _detail {
namespace _option {
struct BinCerealOptionBase {
	VEG_TEMPLATE(
			(typename T, typename File),
			requires(VEG_CONCEPT(bin_cereal<T, File>)),
			static void serialize_to,
			(f, RefMut<File>),
			(t, Ref<Option<T>>))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(aux::cereal::xnothrow_bin_serializable<T, File>)) {
		bool is_some = t.get().is_some();
		cereal::BinCereal<bool>::serialize_to(VEG_FWD(f), ref(is_some));
		if (is_some) {
			cereal::BinCereal<T>::serialize_to(
					VEG_FWD(f),
					t //
							.get()
							.as_ref()
							.unwrap_unchecked(unsafe));
		}
	}
};
struct DbgOptionBase {
	template <typename T>
	static void to_string(fmt::BufferMut out, Ref<Option<T>> opt) {
		if (opt.get().is_some()) {
			out.append_literal(u8"some(");
			fmt::Debug<T>::to_string(
					VEG_FWD(out), opt.get().as_ref().unwrap_unchecked(unsafe));
			out.append_literal(u8")");
		} else {
			out.append_literal(u8"none");
		}
	}
};
struct OrdOptionBase {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_NODISCARD VEG_INLINE static constexpr auto cmp,
			(lhs, Ref<Option<T>>),
			(rhs, Ref<Option<U>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->cmp::Ordering {
		return (lhs.get().is_some()) //
		           ? rhs.get().is_some()
		                 ? static_cast<cmp::Ordering>(cmp::Ord<T, U>::cmp(
													 lhs.get().as_ref().unwrap_unchecked(unsafe),
													 rhs.get().as_ref().unwrap_unchecked(unsafe)))
		                 : cmp::Ordering::greater
		           // lhs is none
		           : rhs.get().is_none() ? cmp::Ordering::equal
		                                 : cmp::Ordering::less;
	}
};
struct OrdOptionBaseLhsNone {
	template <typename U>
	VEG_NODISCARD VEG_INLINE static constexpr auto
	cmp(Ref<None> /*lhs*/, Ref<Option<U>> rhs) VEG_NOEXCEPT -> cmp::Ordering {
		return rhs.get().is_none() ? cmp::Ordering::equal : cmp::Ordering::less;
	}
};
struct OrdOptionBaseRhsNone {
	template <typename T>
	VEG_NODISCARD VEG_INLINE static constexpr auto
	cmp(Ref<Option<T>> lhs, Ref<None> /*rhs*/) VEG_NOEXCEPT -> cmp::Ordering {
		return lhs.get().is_none() ? cmp::Ordering::equal : cmp::Ordering::greater;
	}
};
} // namespace _option
} // namespace _detail

template <typename T>
struct cereal::BinCereal<Option<T>> : _detail::_option::BinCerealOptionBase {};

template <typename T>
struct fmt::Debug<Option<T>> : _detail::_option::DbgOptionBase {};
template <>
struct fmt::Debug<None> {
	static void to_string(fmt::BufferMut out, Ref<None> /*unused*/) {
		out.append_literal(u8"none");
	}
};
template <typename T, typename U>
struct cmp::Ord<Option<T>, Option<U>> : _detail::_option::OrdOptionBase {};
template <typename U>
struct cmp::Ord<None, Option<U>> : _detail::_option::OrdOptionBaseLhsNone {};
template <typename T>
struct cmp::Ord<Option<T>, None> : _detail::_option::OrdOptionBaseRhsNone {};
template <>
struct cmp::Ord<None, None> : _detail::_option::OrdOptionBaseRhsNone {
	VEG_NODISCARD VEG_INLINE static constexpr auto
	cmp(Ref<None> /*unused*/, Ref<None> /*unused*/) VEG_NOEXCEPT
			-> cmp::Ordering {
		return cmp::Ordering::equal;
	}
};

template <typename T>
struct cpo::is_trivially_constructible<Option<T>>
		: cpo::is_trivially_constructible<T> {};
template <typename T>
struct cpo::is_trivially_relocatable<Option<T>>
		: cpo::is_trivially_relocatable<T> {};
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_OPTION_HPP_8NVLXES2S */
