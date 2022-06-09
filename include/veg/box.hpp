#ifndef VEG_BOX_HPP_HUJDLY6PS
#define VEG_BOX_HPP_HUJDLY6PS

#include "veg/internal/delete_special_members.hpp"
#include "veg/type_traits/constructible.hpp"
#include "veg/memory/alloc.hpp"
#include "veg/tuple.hpp"
#include "veg/util/assert.hpp"
#include "veg/internal/prologue.hpp"

namespace veg {

namespace _detail {
namespace _mem {
template <typename T>
struct UniquePtr {
	T* inner = nullptr;
	UniquePtr() = default;
	VEG_INLINE UniquePtr(UniquePtr&& rhs) noexcept : inner{rhs.inner} {
		rhs.inner = {};
	}
	VEG_INLINE UniquePtr(FromRawParts /*from_raw_parts*/, T* ptr) noexcept
			: inner{ptr} {}

	UniquePtr(UniquePtr const& /*unused*/) = delete;
	auto operator=(UniquePtr const&) -> UniquePtr& = delete;
	auto operator=(UniquePtr&&) -> UniquePtr& = delete;
	~UniquePtr() = default;
};

template <typename T, typename A>
struct BoxAlloc : Tuple<A, UniquePtr<T>> {
	BoxAlloc() = default;

	using Tuple<A, UniquePtr<T>>::Tuple;

	BoxAlloc(BoxAlloc&&) = default;
	BoxAlloc(BoxAlloc const&) = default;
	auto operator=(BoxAlloc&&) -> BoxAlloc& = delete;
	auto operator=(BoxAlloc const&) -> BoxAlloc& = delete;

	VEG_INLINE ~BoxAlloc()
			VEG_NOEXCEPT_IF(VEG_CONCEPT(alloc::nothrow_dealloc<A>)) {
		if ((*this)[1_c].inner != nullptr) {
			mem::Alloc<A>::dealloc(
					mut((*this)[0_c]),
					static_cast<void*>((*this)[1_c].inner),
					mem::Layout{sizeof(T), alignof(T)});
		}
	}
};

template <
		typename T,
		typename A,
		mem::DtorAvailable Dtor,
		mem::CopyAvailable Copy>
struct BoxImpl {
	BoxAlloc<T, A> _;

	VEG_INLINE auto alloc_ref() const noexcept -> Ref<A> { return ref(_[0_c]); }
	VEG_INLINE auto alloc_mut(Unsafe /*tag*/) noexcept -> RefMut<A> {
		return mut(_[0_c]);
	}
	VEG_INLINE auto data_ref() const noexcept -> Ref<T const*> {
		return ref(*static_cast<T const* const*>(mem::addressof(_[1_c].inner)));
	}
	VEG_INLINE auto data_mut(Unsafe /*tag*/) noexcept -> RefMut<T*> {
		return mut(_[1_c].inner);
	}

	VEG_INLINE auto ptr() const noexcept -> T const* { return _[1_c].inner; }
	VEG_INLINE auto ptr_mut() noexcept -> T* { return _[1_c].inner; }

	VEG_INLINE auto operator*() const noexcept -> T const& {
		VEG_ASSERT(ptr() != nullptr);
		return *ptr();
	}
	VEG_INLINE auto operator*() noexcept -> T& {
		VEG_ASSERT(ptr() != nullptr);
		return *ptr_mut();
	}
	VEG_INLINE auto operator->() const noexcept -> T const* {
		VEG_ASSERT(ptr() != nullptr);
		return ptr();
	}
	VEG_INLINE auto operator->() noexcept -> T* {
		VEG_ASSERT(ptr() != nullptr);
		return ptr_mut();
	}

	template <int _>
	VEG_INLINE BoxImpl(
			Unsafe /*unsafe*/, _::FromRawParts<_> /*from_raw_parts*/, A alloc, T* ptr)
			VEG_NOEXCEPT
			: _{
						inplace[tuplify],
						MoveFn<A>{VEG_FWD(alloc)},
						VEG_LAZY_BY_REF(_detail::_mem::UniquePtr<T>{from_raw_parts, ptr}),
				} {}

	template <int _>
	VEG_INLINE BoxImpl(_::FromAlloc<_> /*from_alloc*/, A alloc) VEG_NOEXCEPT
			: _{
						inplace[tuplify],
						MoveFn<A>{VEG_FWD(alloc)},
						DefaultFn<UniquePtr<T>>{},
				} {}

private:
	template <typename Fn>
	void _emplace_no_dtor_unchecked(Fn fn)
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_fn_once<Fn, T>)) {
		mem::Layout l{
				sizeof(T),
				alignof(T),
		};

		_detail::_mem::ManagedAlloc<A> block{
				mem::Alloc<A>::alloc(this->alloc_mut(unsafe), l).data,
				l,
				this->alloc_mut(unsafe),
		};

		mem::construct_with(static_cast<T*>(block.data), VEG_FWD(fn));

		this->data_mut(unsafe).get() = static_cast<T*>(block.data);
		block.data = nullptr;
	}

public:
	VEG_TEMPLATE(
			(typename _, typename Fn),
			requires(VEG_CONCEPT(same<_, FromAllocAndValue>)),
			BoxImpl,
			(/*tag*/, InPlace<_>),
			(alloc, A),
			(fn, Fn))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_fn_once<Fn, T>) &&
			VEG_CONCEPT(alloc::nothrow_alloc<A>))
			: _{
						inplace[tuplify],
						MoveFn<A>{VEG_FWD(alloc)},
						DefaultFn<UniquePtr<T>>{},
				} {
		_emplace_no_dtor_unchecked(VEG_FWD(fn));
	}

	template <int _, typename U = T>
	BoxImpl(_::FromAllocAndValue<_> /*tag*/, A alloc, T value) VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_movable<U>) && VEG_CONCEPT(alloc::nothrow_alloc<A>))
			: _{
						inplace[tuplify],
						MoveFn<A>{VEG_FWD(alloc)},
						DefaultFn<UniquePtr<T>>{},
				} {
		_emplace_no_dtor_unchecked(MoveFn<T>{VEG_FWD(value)});
	}

	BoxImpl() = default;
	BoxImpl(BoxImpl&&) = default;
	explicit BoxImpl(BoxImpl const& rhs) VEG_NOEXCEPT_IF(
			VEG_CONCEPT(nothrow_copyable<A>) &&
			VEG_CONCEPT(alloc::nothrow_alloc<A>) &&
			Copy == mem::CopyAvailable::yes_nothrow)
			: _{
						inplace[tuplify],
						CopyFn<A>{rhs.alloc_ref().get()},
						DefaultFn<UniquePtr<T>>{},
				} {
		static_assert(Copy == mem::CopyAvailableFor<T>::value, ".");
		if (rhs.ptr() != nullptr) {
			_emplace_no_dtor_unchecked(CopyFn<T>{*rhs.ptr()});
		}
	}

	VEG_INLINE auto operator=(BoxImpl&& rhs) noexcept -> BoxImpl& {
		{ auto cleanup = static_cast<decltype(rhs)>(*this); }
		this->alloc_mut(unsafe).get() =
				static_cast<A&&>(rhs.alloc_mut(unsafe).get());
		this->data_mut(unsafe).get() = rhs.ptr_mut();

		rhs.data_mut(unsafe).get() = nullptr;
		return *this;
	}

	auto operator=(BoxImpl const& rhs) VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_alloc<A>) &&
			VEG_CONCEPT(nothrow_copy_assignable<A>) &&
			Copy == mem::CopyAvailable::yes_nothrow) -> BoxImpl& {
		static_assert(Copy == mem::CopyAvailableFor<T>::value, ".");
		if (this != mem::addressof(rhs)) {
			if (cmp::eq(this->alloc_ref(), rhs.alloc_ref()) && //
			    ptr() != nullptr && rhs.ptr() != nullptr) {
				alloc_mut(unsafe).get() = rhs.alloc_ref().get();
				*ptr_mut() = *rhs.ptr();
			} else {
				*this = BoxImpl(rhs);
			}
		}
		return *this;
	}

	VEG_INLINE ~BoxImpl() VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_dealloc<A>) &&
			Dtor == mem::DtorAvailable::yes_nothrow) {
		static_assert(Dtor == mem::DtorAvailableFor<T>::value, ".");

		auto ptr = this->ptr_mut();
		if (ptr != nullptr) {
			mem::destroy_at(ptr);
		}
	}
};
} // namespace _mem
} // namespace _detail

namespace _mem {
namespace _boxadl {
struct AdlBase {};
} // namespace _boxadl
} // namespace _mem

template <
		typename T,
		typename A = mem::SystemAlloc,
		mem::DtorAvailable Dtor = mem::DtorAvailableFor<T>::value,
		mem::CopyAvailable Copy = mem::CopyAvailableFor<T>::value>
struct Box : private _mem::_boxadl::AdlBase,
						 private meta::if_t< //
								 Copy == mem::CopyAvailable::no,
								 _detail::NoCopy,
								 _detail::Empty>,
						 public _detail::_mem::BoxImpl<T, A, Dtor, Copy> {
private:
	using Base = _detail::_mem::BoxImpl<T, A, Dtor, Copy>;

public:
	using Base::Base;
};

namespace _mem {
namespace _boxadl {
VEG_TEMPLATE(
		(typename LT, typename RT, typename LA, typename RA),
		requires(VEG_CONCEPT(eq<LT, RT>)),
		VEG_NODISCARD auto
		operator==,
		(lhs, Box<LT, LA> const&),
		(rhs, Box<RT, RA> const&))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<LT, RT>))->bool {
	return (lhs.ptr() == nullptr || rhs.ptr() == nullptr)
	           ? (lhs.ptr() == nullptr && rhs.ptr() == nullptr)
	           : (*lhs.ptr() == *rhs.ptr());
}
} // namespace _boxadl
} // namespace _mem
namespace mem {} // namespace mem

namespace _detail {
namespace _mem {
struct OrdBox {
	VEG_TEMPLATE(
			(typename LT, typename RT, typename LA, typename RA),
			requires(VEG_CONCEPT(ord<LT, RT>)),
			VEG_NODISCARD static auto cmp,
			(lhs, Ref<Box<LT, LA>>),
			(rhs, Ref<Box<RT, RA>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<LT, RT>))->cmp::Ordering {
		if (lhs.get().ptr() == nullptr && rhs.get().ptr() == nullptr) {
			return cmp::Ordering::equal;
		}
		if (lhs.get().ptr() == nullptr) {
			return cmp::Ordering::less;
		}
		if (rhs.get().ptr() == nullptr) {
			return cmp::Ordering::greater;
		}
		return cmp::cmp( //
				ref(*lhs.get().ptr()),
				ref(*rhs.get().ptr()));
	}
};
struct DbgBox {
	template <typename T>
	static void to_string_impl(fmt::BufferMut out, T const* ptr) noexcept {
		if (ptr != nullptr) {
			out.append_literal(u8"some(");
			fmt::dbg_to(VEG_FWD(out), ref(*ptr));
			out.append_literal(u8")");
		} else {
			out.append_literal(u8"none");
		}
	}

	template <typename T, typename A>
	static void to_string(fmt::BufferMut out, Ref<Box<T, A>> r) noexcept {
		to_string_impl(VEG_FWD(out), r.get().ptr());
	}
};
} // namespace _mem
} // namespace _detail

template <typename T, typename A>
struct cpo::is_trivially_relocatable<Box<T, A>>
		: cpo::is_trivially_relocatable<A> {};
template <typename T, typename A>
struct cpo::is_trivially_constructible<Box<T, A>>
		: cpo::is_trivially_constructible<A> {};

template <typename LT, typename RT, typename LA, typename RA>
struct cmp::Ord<Box<LT, LA>, Box<RT, RA>> : _detail::_mem::OrdBox {};

template <typename T, typename A>
struct fmt::Debug<Box<T, A>> : _detail::_mem::DbgBox {};

namespace nb {
struct box {
	template <typename T>
	VEG_INLINE auto operator()(T val) const
			VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> Box<T> {
		return {
				from_alloc_and_value,
				mem::SystemAlloc{},
				VEG_FWD(val),
		};
	}
};

struct box_with_alloc {
	VEG_TEMPLATE(
			(typename A, typename T),
			requires(
					VEG_CONCEPT(nothrow_movable<A>) && //
					VEG_CONCEPT(nothrow_move_assignable<A>)),
			VEG_INLINE auto
			operator(),
			(alloc, A))
	VEG_NOEXCEPT->Box<T, A> {
		return {
				from_alloc,
				VEG_FWD(alloc),
		};
	}
};

struct box_with_alloc_and_value {
	VEG_TEMPLATE(
			(typename A, typename T),
			requires(
					VEG_CONCEPT(nothrow_movable<A>) &&         //
					VEG_CONCEPT(nothrow_move_assignable<A>) && //
					VEG_CONCEPT(alloc::alloc<A>)),
			VEG_INLINE auto
			operator(),
			(alloc, A),
			(val, T))
	VEG_NOEXCEPT_IF(
			VEG_CONCEPT(alloc::nothrow_alloc<A>) && //
			VEG_CONCEPT(nothrow_movable<T>))
			->Box<T, A> {
		return {
				from_alloc_and_value,
				VEG_FWD(alloc),
				VEG_FWD(val),
		};
	}
};
} // namespace nb
VEG_NIEBLOID(box);
VEG_NIEBLOID(box_with_alloc);
VEG_NIEBLOID(box_with_alloc_and_value);
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_BOX_HPP_HUJDLY6PS */
