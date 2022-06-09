#ifndef VEG_SLICE_HPP_GKSTE2JDS
#define VEG_SLICE_HPP_GKSTE2JDS

#include "veg/util/assert.hpp"
#include "veg/util/get.hpp"
#include "veg/internal/narrow.hpp"
#include "veg/tuple.hpp"
#include "veg/util/compare.hpp"
#include "veg/internal/prologue.hpp"
#include <initializer_list>

namespace veg {
template <typename T, usize N>
using CArray = T[N];

namespace _detail {
namespace _slice {
namespace adl {
struct AdlBase {};
} // namespace adl
} // namespace _slice
} // namespace _detail

template <typename T>
struct Slice : _detail::_slice::adl::AdlBase {
private:
	T const* data = nullptr;
	isize size = 0;

public:
	VEG_INLINE
	constexpr Slice() = default;

	VEG_INLINE
	constexpr Slice(
			Unsafe /*tag*/, FromRawParts /*tag*/, T const* data_, isize count)
			VEG_NOEXCEPT : data{data_},
										 size{count} {}

	VEG_NODISCARD
	VEG_INLINE
	constexpr auto ptr() const VEG_NOEXCEPT -> T const* { return data; }
	VEG_NODISCARD
	VEG_INLINE
	constexpr auto len() const VEG_NOEXCEPT -> isize { return size; }

	VEG_NODISCARD
	VEG_INLINE
	constexpr auto operator[](isize idx) const VEG_NOEXCEPT -> T const& {
		return VEG_INTERNAL_ASSERT_PRECONDITION(usize(idx) < usize(len())),
		       *(data + idx);
	}
	VEG_NODISCARD
	VEG_INLINE
	constexpr auto get_unchecked(Unsafe /*tag*/, isize idx) const VEG_NOEXCEPT
			-> Ref<T> {
		return ref(*(data + idx));
	}

	VEG_NODISCARD VEG_INLINE constexpr auto split_at(isize idx) const VEG_NOEXCEPT
			-> Tuple<Slice<T>, Slice<T>> {
		return VEG_INTERNAL_ASSERT_PRECONDITION(usize(idx) <= usize(len())),
		       Tuple<Slice<T>, Slice<T>>{
							 tuplify,
							 Slice<T>{
									 unsafe,
									 FromRawParts{},
									 data,
									 idx,
							 },
							 Slice<T>{
									 unsafe,
									 FromRawParts{},
									 data + idx,
									 size - idx,
							 },
					 };
	}

	VEG_NODISCARD VEG_INLINE auto as_bytes() const VEG_NOEXCEPT
			-> Slice<unsigned char> {
		return {
				unsafe,
				from_raw_parts,
				reinterpret_cast<unsigned char const*>(data),
				isize(sizeof(T)) * size,
		};
	}
};

template <typename T>
struct SliceMut : private Slice<T> {
	VEG_INLINE
	constexpr SliceMut() = default;

	VEG_INLINE
	constexpr SliceMut(
			Unsafe /*tag*/, FromRawParts /*tag*/, T const* data_, isize count)
			VEG_NOEXCEPT : Slice<T>{
												 unsafe,
												 from_raw_parts,
												 data_,
												 count,
										 } {}

	using Slice<T>::ptr;
	using Slice<T>::as_bytes;
	using Slice<T>::split_at;
	using Slice<T>::len;
	using Slice<T>::get_unchecked;

	VEG_NODISCARD VEG_INLINE constexpr auto as_const() const noexcept
			-> Slice<T> {
		return *this;
	}

	VEG_NODISCARD
	VEG_INLINE
	VEG_CPP14(constexpr) auto operator[](isize idx) VEG_NOEXCEPT -> T& {
		return const_cast<T&>(static_cast<Slice<T> const&>(*this)[idx]);
	}
	VEG_NODISCARD
	VEG_INLINE
	VEG_CPP14(constexpr) auto ptr_mut() VEG_NOEXCEPT -> T* {
		return const_cast<T*>(ptr());
	}
	VEG_NODISCARD
	VEG_INLINE
	VEG_CPP14(constexpr)
	auto get_mut_unchecked(Unsafe /*tag*/, isize idx) VEG_NOEXCEPT -> RefMut<T> {
		return mut(const_cast<T&>(*(this->data + idx)));
	}
	VEG_NODISCARD VEG_INLINE auto as_mut_bytes() VEG_NOEXCEPT
			-> SliceMut<unsigned char> {
		return {
				unsafe,
				from_raw_parts,
				reinterpret_cast<unsigned char*>(ptr_mut()),
				isize(sizeof(T)) * len(),
		};
	}

	VEG_NODISCARD VEG_INLINE VEG_CPP14(constexpr) auto split_at_mut(isize idx)
			VEG_NOEXCEPT -> Tuple<SliceMut<T>, SliceMut<T>> {
		return VEG_INTERNAL_ASSERT_PRECONDITION(usize(idx) <= usize(len())),
		       Tuple<SliceMut<T>, SliceMut<T>>{
							 tuplify,
							 SliceMut<T>{
									 unsafe,
									 from_raw_parts,
									 ptr_mut(),
									 idx,
							 },
							 SliceMut<T>{
									 unsafe,
									 from_raw_parts,
									 ptr_mut() + idx,
									 len() - idx,
							 },
					 };
	}
};

namespace array {
template <typename T, isize N>
struct Array {
	static_assert(N > 0, ".");
	T _[usize{N}];

	constexpr auto as_ref() const -> Slice<T> {
		return {
				unsafe,
				from_raw_parts,
				static_cast<T const*>(_),
				N,
		};
	}
	VEG_CPP14(constexpr) auto as_mut() -> SliceMut<T> {
		return {
				unsafe,
				from_raw_parts,
				static_cast<T*>(_),
				N,
		};
	}
};
} // namespace array
using array::Array;

namespace _detail {
namespace _slice {
namespace adl {
VEG_TEMPLATE(
		(typename T, typename U),
		requires(VEG_CONCEPT(eq<T, U>)),
		VEG_NODISCARD static VEG_CPP14(constexpr) auto
		operator==,
		(lhs, Slice<T>),
		(rhs, Slice<U>))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
	if (lhs.len() != rhs.len()) {
		return false;
	}
	for (isize i = 0; i < lhs.len(); ++i) {
		if (!(lhs.get_unchecked(unsafe, i).get() ==
		      rhs.get_unchecked(unsafe, i).get())) {
			return false;
		}
	}
	return true;
};
VEG_TEMPLATE(
		(typename T, typename U),
		requires(VEG_CONCEPT(eq<T, U>)),
		VEG_NODISCARD static VEG_CPP14(constexpr) auto
		operator==,
		(lhs, SliceMut<T>),
		(rhs, SliceMut<U>))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
	return adl::operator==(lhs.as_const(), rhs.as_const());
};
} // namespace adl

struct DbgSliceBase {
	template <typename T>
	static void to_string(fmt::BufferMut out, Ref<Slice<T>> arg) {
		T const* ptr = arg.get().ptr();
		isize len = arg.get().len();

		_detail::_fmt::DbgStructScope _{out};
		for (isize i = 0; i < len; ++i) {
			out.append_ln();
			fmt::Debug<T>::to_string(out, ref(ptr[i]));
			out.append_literal(u8",");
		}
	}
};
struct DbgSliceMutBase {
	template <typename T>
	static void to_string(fmt::BufferMut out, Ref<SliceMut<T>> arg) {
		DbgSliceBase::to_string(VEG_FWD(out), ref((Slice<T> const&)(arg.get())));
	}
};
struct DbgArrayBase {
	template <typename T, isize N>
	static void to_string(fmt::BufferMut out, Ref<Array<T, N>> arg) {
		DbgSliceBase::to_string(VEG_FWD(out), ref(arg.get().as_ref()));
	}
};

struct OrdSliceBase {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_NODISCARD static VEG_CPP14(constexpr) auto cmp,
			(lhs, Ref<Slice<T>>),
			(rhs, Ref<Slice<U>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->cmp::Ordering {
		Slice<T> lhs_ = lhs.get();
		Slice<U> rhs_ = rhs.get();

		isize common_len = lhs_.len() < rhs_.len() ? lhs_.len() : rhs_.len();
		for (isize i = 0; i < common_len; ++i) {
			auto const val = static_cast<cmp::Ordering>(cmp::Ord<T, U>::cmp( //
					lhs_.get_unchecked(unsafe, i),
					rhs_.get_unchecked(unsafe, i)));
			if (val != cmp::Ordering::equal) {
				return val;
			}
		}
		return cmp::Ord<isize, isize>::cmp(ref(lhs_.len()), ref(rhs_.len()));
	}
};
struct OrdSliceMutBase {
	VEG_TEMPLATE(
			(typename T, typename U),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_NODISCARD static VEG_CPP14(constexpr) auto cmp,
			(lhs, Ref<SliceMut<T>>),
			(rhs, Ref<SliceMut<U>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->cmp::Ordering {
		return OrdSliceBase::cmp( //
				ref(lhs.get().as_const()),
				ref(rhs.get().as_const()));
	}
};
struct OrdArrayBase {
	VEG_TEMPLATE(
			(typename T, isize N, typename U, isize M),
			requires(VEG_CONCEPT(ord<T, U>)),
			VEG_NODISCARD static VEG_CPP14(constexpr) auto cmp,
			(lhs, Ref<Array<T, N>>),
			(rhs, Ref<Array<U, M>>))
	VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_ord<T, U>))->cmp::Ordering {
		return OrdSliceBase::cmp( //
				ref(lhs.get().as_ref()),
				ref(rhs.get().as_ref()));
	}
};
} // namespace _slice
} // namespace _detail
namespace array {
VEG_TEMPLATE(
		(typename T, isize N, typename U, isize M),
		requires(VEG_CONCEPT(eq<T, U>)),
		VEG_NODISCARD static VEG_CPP14(constexpr) auto
		operator==,
		(lhs, Array<T, N> const&),
		(rhs, Array<U, M> const&))
VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_eq<T, U>))->bool {
	return (N == M) &&
	       _detail::_slice::adl::operator==(lhs.as_ref(), rhs.as_ref());
}
} // namespace array

namespace nb {
struct init_list {
	template <typename T>
	VEG_CPP14(constexpr)
	auto operator()(std::initializer_list<T> init_list) const noexcept
			-> Slice<T> {
		return {
				unsafe,
				from_raw_parts,
				init_list.begin(),
				isize(init_list.size()),
		};
	}
};
} // namespace nb
VEG_NIEBLOID(init_list);

template <typename T>
struct cpo::is_trivially_constructible<Slice<T>> : meta::bool_constant<true> {};

template <typename T, typename U>
struct cmp::Ord<Slice<T>, Slice<U>> : _detail::_slice::OrdSliceBase {};
template <typename T, typename U>
struct cmp::Ord<SliceMut<T>, SliceMut<U>> : _detail::_slice::OrdSliceMutBase {};
template <typename T, isize N, typename U, isize M>
struct cmp::Ord<Array<T, N>, Array<U, M>> : _detail::_slice::OrdArrayBase {};

template <typename T>
struct fmt::Debug<Slice<T>> : _detail::_slice::DbgSliceBase {};
template <typename T>
struct fmt::Debug<SliceMut<T>> : _detail::_slice::DbgSliceMutBase {};
template <typename T, isize N>
struct fmt::Debug<Array<T, N>> : _detail::_slice::DbgArrayBase {};
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_SLICE_HPP_GKSTE2JDS */
