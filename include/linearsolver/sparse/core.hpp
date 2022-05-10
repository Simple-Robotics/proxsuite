#ifndef SPARSE_LDLT_CORE_HPP_TL3ADOGVS
#define SPARSE_LDLT_CORE_HPP_TL3ADOGVS

#include <veg/slice.hpp>
#include <veg/memory/dynamic_stack.hpp>
#include <type_traits>
#include <Eigen/SparseCore>

#define SPARSE_LDLT_CONCEPT(...)                                               \
	VEG_CONCEPT_MACRO(::linearsolver::sparse::concepts, __VA_ARGS__)
#define SPARSE_LDLT_CHECK_CONCEPT(...)                                         \
	VEG_CONCEPT_MACRO(::linearsolver::sparse::concepts, __VA_ARGS__)

namespace linearsolver {
namespace sparse {
using veg::dynstack::DynStackMut;
using namespace veg::literals;

using veg::usize;
using veg::isize;

using veg::Slice;
using veg::SliceMut;

using veg::Ref;
using veg::RefMut;
using veg::ref;
using veg::mut;

inline namespace tags {
using veg::Unsafe;
using veg::unsafe;

using veg::FromRawParts;
using veg::from_raw_parts;
VEG_TAG(from_eigen, FromEigen);
} // namespace tags

namespace concepts {} // namespace concepts

namespace _detail {
template <typename I, bool = sizeof(I) < sizeof(int)>
struct WrappingPlusType;

template <typename I>
struct WrappingPlusType<I, true> {
	using Promoted = unsigned;
};
template <typename I>
struct WrappingPlusType<I, false> {
	using Promoted = typename std::make_unsigned<I>::type;
};
} // namespace _detail
namespace util {
namespace nb {
struct wrapping_plus {
	template <typename I>
	auto operator()(I a, I b) const noexcept -> I {
		using U = typename _detail::WrappingPlusType<I>::Promoted;
		return I(U(a) + U(b));
	}
};
struct checked_non_negative_plus {
	template <typename I>
	auto operator()(I a, I b) const noexcept -> I {
		return (
				VEG_ASSERT(wrapping_plus{}(a, b) >= a), //
				wrapping_plus{}(a, b));
	}
};

struct wrapping_inc {
	template <typename I>
	auto operator()(RefMut<I> a) const noexcept -> I {
		return a.get() = wrapping_plus{}(a.get(), I(1));
	}
};
struct wrapping_dec {
	template <typename I>
	auto operator()(RefMut<I> a) const noexcept -> I {
		return a.get() = wrapping_plus{}(a.get(), I(-1));
	}
};
struct sign_extend {
	template <typename I>
	auto operator()(I a) const noexcept -> usize {
		return usize(isize(typename std::make_signed<I>::type(a)));
	}
};
struct zero_extend {
	template <typename I>
	auto operator()(I a) const noexcept -> usize {
		return usize(typename std::make_unsigned<I>::type(a));
	}
};
} // namespace nb
VEG_NIEBLOID(wrapping_plus);
VEG_NIEBLOID(checked_non_negative_plus);
VEG_NIEBLOID(wrapping_inc);
VEG_NIEBLOID(wrapping_dec);
VEG_NIEBLOID(sign_extend);
VEG_NIEBLOID(zero_extend);
} // namespace util

template <typename T>
struct DenseVecRef {
	DenseVecRef() = default;
	DenseVecRef(FromRawParts /*from_raw_parts*/, Slice<T> data) noexcept
			: _{
						data.ptr(),
						data.len(),
				} {}
	template <typename V>
	DenseVecRef(FromEigen /*from_eigen*/, V const& v) noexcept
			: _{
						v.data(),
						v.rows(),
				} {
		static_assert(V::InnerStrideAtCompileTime == 1, ".");
		static_assert(V::ColsAtCompileTime == 1, ".");
	}

	auto as_slice() const noexcept -> Slice<T> {
		return {
				unsafe,
				from_raw_parts,
				_.ptr,
				_.size,
		};
	}
	auto nrows() const noexcept -> isize { return _.size; }
	auto ncols() const noexcept -> isize { return 1; }

	auto to_eigen() const noexcept -> Eigen::Map<Eigen::Matrix<T, -1, 1> const> {
		return {_.ptr, _.size};
	}

private:
	struct {
		T const* ptr;
		isize size;
	} _ = {};
};

template <typename T>
struct DenseVecMut {
	DenseVecMut() = default;
	DenseVecMut(FromRawParts /*from_raw_parts*/, SliceMut<T> data) noexcept
			: _{
						data.ptr_mut(),
						data.len(),
				} {}
	template <typename V>
	DenseVecMut(FromEigen /*from_eigen*/, V&& v) noexcept
			: _{
						v.data(),
						v.rows(),
				} {
		static_assert(veg::uncvref_t<V>::InnerStrideAtCompileTime == 1, ".");
		static_assert(veg::uncvref_t<V>::ColsAtCompileTime == 1, ".");
	}

	auto as_slice() const noexcept -> Slice<T> {
		return {
				unsafe,
				from_raw_parts,
				_.ptr,
				_.size,
		};
	}
	auto as_slice_mut() noexcept -> SliceMut<T> {
		return {
				unsafe,
				from_raw_parts,
				_.ptr,
				_.size,
		};
	}

	auto as_const() const noexcept -> DenseVecRef<T> {
		return {from_raw_parts, as_slice()};
	}
	auto nrows() const noexcept -> isize { return _.size; }
	auto ncols() const noexcept -> isize { return 1; }

	auto to_eigen() const noexcept -> Eigen::Map<Eigen::Matrix<T, -1, 1>> {
		return {_.ptr, _.size};
	}

private:
	struct {
		T* ptr;
		isize size;
	} _ = {};
};

template <typename T, typename I = isize>
struct VecRef {
	VecRef( //
			FromRawParts /*from_raw_parts*/,
			isize nrows,
			Slice<I> row_indices,
			Slice<T> values)
			: _{
						(VEG_ASSERT(row_indices.len() == values.len()),
	           decltype(_){
								 nrows,
								 row_indices.len(),
								 row_indices.ptr(),
								 values.ptr(),
						 }),
				} {}

	auto nrows() const noexcept -> isize { return _.nrows; }
	auto ncols() const noexcept -> isize { return 1; }
	auto nnz() const noexcept -> isize { return _.nnz; }

	auto row_indices() const noexcept -> Slice<I> {
		return {unsafe, from_raw_parts, _.row, nnz()};
	}
	auto values() const noexcept -> Slice<T> {
		return {unsafe, from_raw_parts, _.val, nnz()};
	}

private:
	struct {
		isize nrows;
		isize nnz;
		I const* row;
		T const* val;
	} _;
};

template <typename T, typename I = isize>
struct VecMut {
	VecMut( //
			FromRawParts /*from_raw_parts*/,
			isize nrows,
			Slice<I> row_indices,
			Slice<T> values)
			: _{
						(VEG_ASSERT(row_indices.len() == values.len()),
	           decltype(_){
								 nrows,
								 row_indices.ptr_mut(),
								 values.ptr_mut(),
						 }),
				} {}

	auto nrows() const noexcept -> isize { return _.nrows; }
	auto ncols() const noexcept -> isize { return 1; }
	auto nnz() const noexcept -> isize { return _.nnz; }

	auto row_indices() const noexcept -> Slice<I> {
		return {unsafe, from_raw_parts, _.row, nnz()};
	}
	auto values() const noexcept -> Slice<T> {
		return {unsafe, from_raw_parts, _.val, nnz()};
	}
	auto row_indices_mut() noexcept -> SliceMut<I> {
		return {unsafe, from_raw_parts, _.row, nnz()};
	}
	auto values_mut() noexcept -> SliceMut<T> {
		return {unsafe, from_raw_parts, _.val, nnz()};
	}

	auto as_const() const noexcept -> VecRef<T, I> {
		return {from_raw_parts, nrows(), row_indices(), values()};
	}

private:
	struct {
		isize nrows;
		isize nnz;
		I* row;
		T* val;
	} _;
};

namespace _detail {
template <typename D, typename I>
struct SymbolicMatRefInterface {
private:
	template <typename U = D>
	auto _() const noexcept -> decltype((VEG_DECLVAL(U const&)._)) {
		return static_cast<D const*>(this)->_;
	}

public:
	auto nrows() const noexcept -> isize { return _().nrows; }
	auto ncols() const noexcept -> isize { return _().ncols; }
	auto nnz() const noexcept -> isize { return _().nnz; }
	auto max_nnz() const noexcept -> isize { return _().max_nnz; }

	auto col_ptrs() const noexcept -> Slice<I> {
		return {unsafe, from_raw_parts, _().col, ncols() + 1};
	}
	auto row_indices() const noexcept -> Slice<I> {
		return {unsafe, from_raw_parts, _().row, max_nnz()};
	}
	auto nnz_per_col() const noexcept -> Slice<I> {
		return {
				unsafe,
				from_raw_parts,
				_().nnz_per_col,
				_().nnz_per_col == nullptr ? isize(0) : ncols(),
		};
	}
	auto is_compressed() const noexcept -> bool {
		return _().nnz_per_col == nullptr;
	}
	auto col_start(usize j) const noexcept -> usize {
		return VEG_ASSERT(j < ncols()), util::zero_extend(_().col[j]);
	}
	auto col_start_unchecked(Unsafe /*unsafe*/, usize j) const noexcept -> usize {
		return VEG_DEBUG_ASSERT(j < ncols()), util::zero_extend(_().col[j]);
	}
	auto col_end(usize j) const noexcept -> usize {
		return VEG_ASSERT(j < ncols()), col_end_unchecked(unsafe, j);
	}
	auto col_end_unchecked(Unsafe /*unsafe*/, usize j) const noexcept -> usize {
		return VEG_DEBUG_ASSERT(j < ncols()),
		       util::zero_extend(
							 is_compressed() ? _().col[j + 1]
															 : I(_().col[j] + _().nnz_per_col[j]));
	}
};
template <typename D, typename I>
struct SymbolicMatMutInterface : SymbolicMatRefInterface<D, I> {
private:
	template <typename U = D>
	auto _() noexcept -> decltype((VEG_DECLVAL(U&)._)) {
		return static_cast<D*>(this)->_;
	}

public:
	auto col_ptrs_mut() noexcept -> SliceMut<I> {
		return {unsafe, from_raw_parts, _().col, this->ncols() + 1};
	}
	auto row_indices_mut() noexcept -> SliceMut<I> {
		return {unsafe, from_raw_parts, _().row, this->max_nnz()};
	}
	auto nnz_per_col_mut() noexcept -> SliceMut<I> {
		return {
				unsafe,
				from_raw_parts,
				_().nnz_per_col,
				_().nnz_per_col == nullptr ? isize(0) : this->ncols(),
		};
	}
};
} // namespace _detail

template <typename I = isize>
struct SymbolicMatRef : _detail::SymbolicMatRefInterface<SymbolicMatRef<I>, I> {
	friend struct _detail::SymbolicMatRefInterface<SymbolicMatRef, I>;
	SymbolicMatRef(
			FromRawParts /*from_raw_parts*/,
			isize nrows,
			isize ncols,
			isize nnz,
			Slice<I> col_ptrs,
			Slice<I> row_indices,
			Slice<I> nnz_per_col)
			: _{
						(VEG_ASSERT_ALL_OF( //
								 nrows >= 0,
								 ncols >= 0,
								 col_ptrs.len() == ncols + 1,
								 ((nnz_per_col.len() == ncols ||
	                 nnz_per_col.ptr() == nullptr))),
	           decltype(_){
								 nrows,
								 ncols,
								 nnz,
								 row_indices.len(),
								 col_ptrs.ptr(),
								 row_indices.ptr(),
								 nnz_per_col.ptr(),
						 }),
				} {}

private:
	struct {
		isize nrows;
		isize ncols;
		isize nnz;
		isize max_nnz;
		I const* col;
		I const* row;
		I const* nnz_per_col;
	} _;
};
template <typename I = isize>
struct SymbolicMatMut : _detail::SymbolicMatMutInterface<SymbolicMatMut<I>, I> {
	friend struct _detail::SymbolicMatRefInterface<SymbolicMatMut, I>;
	friend struct _detail::SymbolicMatMutInterface<SymbolicMatMut, I>;
	SymbolicMatMut(
			FromRawParts /*from_raw_parts*/,
			isize nrows,
			isize ncols,
			isize nnz,
			SliceMut<I> col_ptrs,
			SliceMut<I> row_indices,
			SliceMut<I> nnz_per_col)
			: _{
						(VEG_ASSERT_ALL_OF( //
								 nrows >= 0,
								 ncols >= 0,
								 col_ptrs.len() == ncols + 1,
								 ((nnz_per_col.len() == ncols ||
	                 nnz_per_col.ptr() == nullptr))),
	           decltype(_){
								 nrows,
								 ncols,
								 nnz,
								 row_indices.len(),
								 col_ptrs.ptr_mut(),
								 row_indices.ptr_mut(),
								 nnz_per_col.ptr_mut(),
						 }),
				} {}

	auto as_const() const noexcept -> SymbolicMatRef<I> {
		return {
				from_raw_parts,
				this->nrows(),
				this->ncols(),
				this->nnz(),
				this->col_ptrs(),
				this->row_indices(),
				this->nnz_per_col(),
		};
	}

private:
	struct {
		isize nrows;
		isize ncols;
		isize nnz;
		isize max_nnz;
		I* col;
		I* row;
		I* nnz_per_col;
	} _;
};

template <typename T, typename I = isize>
struct MatRef : _detail::SymbolicMatRefInterface<MatRef<T, I>, I> {
	friend struct _detail::SymbolicMatRefInterface<MatRef, I>;
	MatRef(
			FromRawParts /*from_raw_parts*/,
			isize nrows,
			isize ncols,
			isize nnz,
			Slice<I> col_ptrs,
			Slice<I> row_indices,
			Slice<I> nnz_per_col,
			Slice<T> values)
			: _{
						(VEG_ASSERT_ALL_OF( //
								 nrows >= 0,
								 ncols >= 0,
								 col_ptrs.len() == ncols + 1,
								 ((nnz_per_col.len() == ncols || nnz_per_col.ptr() == nullptr)),
								 row_indices.len() == values.len()),
	           decltype(_){
								 nrows,
								 ncols,
								 nnz,
								 row_indices.len(),
								 col_ptrs.ptr(),
								 row_indices.ptr(),
								 nnz_per_col.ptr(),
								 values.ptr(),
						 }),
				} {}

	template <typename M>
	MatRef(FromEigen /*from_eigen*/, M const& m)
			: _{
						m.rows(),
						m.cols(),
						m.nonZeros(),
						m.outerIndexPtr()[m.cols()],
						m.outerIndexPtr(),
						m.innerIndexPtr(),
						m.innerNonZeroPtr(),
						m.valuePtr(),
				} {
		static_assert(!bool(M::IsRowMajor), ".");
	};

	auto values() const noexcept -> Slice<T> {
		return {unsafe, from_raw_parts, _.val, this->max_nnz()};
	}
	auto symbolic() const noexcept -> SymbolicMatRef<I> {
		return {
				from_raw_parts,
				this->nrows(),
				this->ncols(),
				this->nnz(),
				this->col_ptrs(),
				this->row_indices(),
				this->nnz_per_col(),
		};
	}

	auto to_eigen() const noexcept
			-> Eigen::Map<Eigen::SparseMatrix<T, Eigen::ColMajor, I> const> {
		return {_.nrows, _.ncols, _.nnz, _.col, _.row, _.val, _.nnz_per_col};
	}

private:
	struct {
		isize nrows;
		isize ncols;
		isize nnz;
		isize max_nnz;
		I const* col;
		I const* row;
		I const* nnz_per_col;
		T const* val;
	} _;
};

template <typename T, typename I = isize>
struct MatMut : _detail::SymbolicMatMutInterface<MatMut<T, I>, I> {
	friend struct _detail::SymbolicMatRefInterface<MatMut, I>;
	friend struct _detail::SymbolicMatMutInterface<MatMut, I>;
	MatMut(
			FromRawParts /*from_raw_parts*/,
			isize nrows,
			isize ncols,
			isize nnz,
			SliceMut<I> col_ptrs,
			SliceMut<I> row_indices,
			SliceMut<I> nnz_per_col,
			SliceMut<T> values)
			: _{
						(VEG_ASSERT_ALL_OF( //
								 nrows >= 0,
								 ncols >= 0,
								 col_ptrs.len() == ncols + 1,
								 ((nnz_per_col.len() == ncols || nnz_per_col.ptr() == nullptr)),
								 row_indices.len() == values.len()),
	           decltype(_){
								 nrows,
								 ncols,
								 nnz,
								 row_indices.len(),
								 col_ptrs.ptr_mut(),
								 row_indices.ptr_mut(),
								 nnz_per_col.ptr_mut(),
								 values.ptr_mut(),
						 }),
				} {}

	template <typename M>
	MatMut(FromEigen /*from_eigen*/, M&& m)
			: _{
						m.rows(),
						m.cols(),
						m.nonZeros(),
						m.outerIndexPtr()[m.cols()],
						m.outerIndexPtr(),
						m.innerIndexPtr(),
						m.innerNonZeroPtr(),
						m.valuePtr(),
				} {
		static_assert(!bool(veg::uncvref_t<M>::IsRowMajor), ".");
	};

	auto values() const noexcept -> Slice<T> {
		return {unsafe, from_raw_parts, _.val, this->max_nnz()};
	}
	auto values_mut() noexcept -> SliceMut<T> {
		return {unsafe, from_raw_parts, _.val, this->max_nnz()};
	}
	auto is_compressed() const noexcept -> bool {
		return _.nnz_per_col == nullptr;
	}

	auto as_const() const noexcept -> MatRef<T, I> {
		return {
				from_raw_parts,
				this->nrows(),
				this->ncols(),
				this->nnz(),
				this->col_ptrs(),
				this->row_indices(),
				this->nnz_per_col(),
				this->values(),
		};
	}
	auto symbolic() const noexcept -> SymbolicMatRef<I> {
		return {
				from_raw_parts,
				this->nrows(),
				this->ncols(),
				this->nnz(),
				this->col_ptrs(),
				this->row_indices(),
				this->nnz_per_col(),
		};
	}
	auto symbolic_mut() const noexcept -> SymbolicMatRef<I> {
		return {
				from_raw_parts,
				this->nrows(),
				this->ncols(),
				this->nnz(),
				this->col_ptrs_mut(),
				this->row_indices_mut(),
				this->nnz_per_col_mut(),
		};
	}
	auto to_eigen() const noexcept
			-> Eigen::Map<Eigen::SparseMatrix<T, Eigen::ColMajor, I>> {
		return {_.nrows, _.ncols, _.nnz, _.col, _.row, _.val, _.nnz_per_col};
	}
	void _set_nnz(isize new_nnz) noexcept { _.nnz = new_nnz; }

private:
	struct {
		isize nrows;
		isize ncols;
		isize nnz;
		isize max_nnz;
		I* col;
		I* row;
		I* nnz_per_col;
		T* val;
	} _;
};
} // namespace sparse
} // namespace linearsolver

namespace veg {
namespace fmt {
template <typename T, typename I>
struct Debug<linearsolver::sparse::MatRef<T, I>> {
	static void
	to_string(BufferMut out, Ref<linearsolver::sparse::MatRef<T, I>> r) noexcept {
		linearsolver::sparse::MatRef<T, I> a = r.get();
		auto pap = a.col_ptrs().ptr();
		auto pai = a.row_indices().ptr();
		auto pax = a.values().ptr();

		out.append_literal(u8"matrix of size [");
		fmt::dbg_to(out, ref(a.nrows()));
		out.append_literal(u8", ");
		fmt::dbg_to(out, ref(a.ncols()));
		out.append_literal(u8"] {");
		++out.indent_level;

		for (isize j = 0; j < a.ncols(); ++j) {
			auto col_start = a.col_start_unchecked(unsafe, j);
			auto col_end = a.col_end_unchecked(unsafe, j);

			for (isize p = col_start; p < col_end; ++p) {
				auto i = linearsolver::sparse::util::zero_extend(pai[p]);
				out.append_ln();

				out.append_literal(u8"row: ");
				fmt::dbg_to(out, ref(i));
				out.append_literal(u8", col: ");
				fmt::dbg_to(out, ref(j));
				out.append_literal(u8", val: ");
				fmt::dbg_to(out, ref(pax[p]));
			}
		}

		--out.indent_level;
		out.append_ln();
		out.append_literal(u8"}");
	}
};

template <typename T, typename I>
struct Debug<linearsolver::sparse::MatMut<T, I>> {
	static void
	to_string(BufferMut out, Ref<linearsolver::sparse::MatMut<T, I>> r) noexcept {
		dbg_to(out, ref(r.get().as_const()));
	}
};
} // namespace fmt
} // namespace veg
#endif /* end of include guard SPARSE_LDLT_CORE_HPP_TL3ADOGVS */
