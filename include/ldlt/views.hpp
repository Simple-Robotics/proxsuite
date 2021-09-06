#ifndef INRIA_LDLT_VIEWS_HPP_UGNXAQSBS
#define INRIA_LDLT_VIEWS_HPP_UGNXAQSBS

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include <cstdint>
#include <Eigen/Core>
#include <new>

namespace ldlt {
namespace detail {
namespace nb {
struct max2 {
	template <typename T>
	LDLT_INLINE constexpr auto operator()(T const& a, T const& b) const
			-> T const& {
		return a > b ? a : b;
	}
};
struct min2 {
	template <typename T>
	LDLT_INLINE constexpr auto operator()(T a, T b) const -> T {
		return (a < b) ? a : b;
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(max2);
LDLT_DEFINE_NIEBLOID(min2);

constexpr auto round_up(i32 n, i32 k) noexcept -> i32 {
	return (n + k - 1) / k * k;
}

inline auto next_aligned(void* ptr, usize align) noexcept -> void* {
	using BytePtr = unsigned char*;
	using VoidPtr = void*;
	using UPtr = std::uintptr_t;

	UPtr mask = align - 1;
	UPtr iptr = UPtr(ptr);
	UPtr aligned_ptr = (iptr + mask) & ~mask;
	return VoidPtr(BytePtr(ptr) + aligned_ptr - iptr);
}

constexpr usize align_simd = (SIMDE_NATURAL_VECTOR_SIZE / 8U);
constexpr usize align_cacheline = 64;
constexpr usize align_simd_and_cacheline = max2(align_simd, align_cacheline);

template <typename T>
struct SimdCachelineAlignStep {
	static constexpr i32 value =
			i32(max2(usize(1), align_simd_and_cacheline / alignof(T)));
};

template <typename T>
struct UniqueMalloca {
	void* malloca_ptr;
	T* data;
	usize n;

	static constexpr usize align = max2(alignof(T), align_simd_and_cacheline);

	static constexpr usize max_stack_count =
			usize(LDLT_MAX_STACK_ALLOC_SIZE) / sizeof(T);

	static constexpr auto can_alloca(usize count) -> bool {
		return count < max_stack_count;
	}

	LDLT_INLINE UniqueMalloca(void* ptr, usize count)
			: malloca_ptr(ptr),
				data(static_cast<T*>(detail::next_aligned(ptr, align))),
				n(count) {
		new (data) T[n];
	}

	UniqueMalloca(UniqueMalloca const&) = delete;
	UniqueMalloca(UniqueMalloca&&) = delete;
	auto operator=(UniqueMalloca const&) -> UniqueMalloca& = delete;
	auto operator=(UniqueMalloca&&) -> UniqueMalloca& = delete;
	LDLT_INLINE ~UniqueMalloca() {
		for (usize i = 0; i < n; ++i) {
			data[i].~T();
		}
		if (n < max_stack_count) {
#if LDLT_HAS_ALLOCA
			LDLT_FREEA(malloca_ptr);
#endif
		} else {
			std::free(malloca_ptr);
		}
	}
};
} // namespace detail

enum struct Layout : unsigned char {
	colmajor = 0,
	rowmajor = 1,
};

constexpr auto flip_layout(Layout l) noexcept -> Layout {
	return Layout(1 - u32(l));
}

constexpr Layout colmajor = Layout::colmajor;
constexpr Layout rowmajor = Layout::rowmajor;

namespace detail {
template <Layout L>
struct ElementAccess;

template <>
struct ElementAccess<Layout::colmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, i32 row, i32 col, i32 outer_stride) noexcept -> T* {
		return ptr + (usize(row) + usize(col) * usize(outer_stride));
	}

	template <usize N, typename T>
	LDLT_INLINE static auto
	load_col_pack(T const* ptr, i32 /*outer_stride*/) noexcept -> Pack<T, N> {
		return Pack<T, N>::load_unaligned(ptr);
	}
	template <usize N, typename T>
	LDLT_INLINE static void
	store_col_pack(T* ptr, Pack<T, N> pack, i32 /*outer_stride*/) noexcept {
		return pack.store_unaligned(ptr);
	}

	using NextRowStride = Eigen::Stride<0, 0>;
	using NextColStride = Eigen::InnerStride<Eigen::Dynamic>;
	LDLT_INLINE static auto next_row_stride(i32 outer_stride) noexcept
			-> NextRowStride {
		(void)outer_stride;
		return NextRowStride{};
	}
	LDLT_INLINE static auto next_col_stride(i32 outer_stride) noexcept
			-> NextColStride {
		return NextColStride /* NOLINT(modernize-return-braced-init-list) */ (
				outer_stride);
	}

	template <typename T>
	LDLT_INLINE static void
	transpose_if_rowmajor(T* ptr, i32 dim, i32 outer_stride) {
		(void)ptr, (void)dim, (void)outer_stride;
	}
};

template <>
struct ElementAccess<Layout::rowmajor> {
	template <typename T>
	LDLT_INLINE static constexpr auto
	offset(T* ptr, i32 row, i32 col, i32 outer_stride) noexcept -> T* {
		return ptr + (usize(col) + usize(row) * usize(outer_stride));
	}

	template <usize N, typename T>
	LDLT_INLINE static auto load_col_pack(T const* ptr, i32 outer_stride) noexcept
			-> Pack<T, N> {
		return Pack<T, N>::load_gather(ptr, outer_stride);
	}
	template <usize N, typename T>
	LDLT_INLINE static void
	store_col_pack(T* ptr, Pack<T, N> pack, i32 outer_stride) noexcept {
		return pack.store_scatter(ptr, outer_stride);
	}

	using NextColStride = Eigen::Stride<0, 0>;
	using NextRowStride = Eigen::InnerStride<Eigen::Dynamic>;
	LDLT_INLINE static auto next_col_stride(i32 outer_stride) noexcept
			-> NextColStride {
		(void)outer_stride;
		return NextColStride{};
	}
	LDLT_INLINE static auto next_row_stride(i32 outer_stride) noexcept
			-> NextRowStride {
		return NextRowStride /* NOLINT(modernize-return-braced-init-list) */ (
				outer_stride);
	}

	template <typename T>
	LDLT_INLINE static void
	transpose_if_rowmajor(T* ptr, i32 dim, i32 outer_stride) {
		Eigen::Map<                            //
				Eigen::Matrix<                     //
						T,                             //
						Eigen::Dynamic,                //
						Eigen::Dynamic                 //
						>,                             //
				Eigen::Unaligned,                  //
				Eigen::OuterStride<Eigen::Dynamic> //
				>{
				ptr,
				dim,
				dim,
				Eigen::OuterStride<Eigen::Dynamic>(outer_stride),
		}
				.transposeInPlace();
	}
};
} // namespace detail

template <typename Scalar>
struct VectorView {
	Scalar const* data;
	i32 dim;

	LDLT_INLINE auto operator()(i32 index) const noexcept -> Scalar const& {
		return *(data + index);
	}
	LDLT_INLINE auto segment(i32 i, i32 size) const noexcept -> VectorView {
		return {data + i, size};
	}
};

template <typename Scalar>
struct VectorViewMut {
	Scalar* data;
	i32 dim;
	LDLT_INLINE auto as_const() const noexcept -> VectorView<Scalar> {
		return {data, dim};
	}
	LDLT_INLINE auto operator()(i32 index) const noexcept -> Scalar& {
		return *(data + index);
	}
	LDLT_INLINE auto segment(i32 i, i32 size) const noexcept -> VectorViewMut {
		return {data + i, size};
	}
};

template <typename Scalar>
struct StridedVectorView {
	Scalar const* data;
	i32 dim;
	i32 stride;

	LDLT_INLINE auto operator()(i32 index) const noexcept -> Scalar const& {
		return *(data + stride * index);
	}
	LDLT_INLINE auto segment(i32 i, i32 size) const noexcept
			-> StridedVectorView {
		return {
				data + stride * i,
				size,
				stride,
		};
	}
};

template <typename Scalar>
struct StridedVectorViewMut {
	Scalar* data;
	i32 dim;
	i32 stride;

	LDLT_INLINE auto as_const() const noexcept -> StridedVectorView<Scalar> {
		return {data, dim, stride};
	}
	LDLT_INLINE auto operator()(i32 index) const noexcept -> Scalar& {
		return *(data + stride * index);
	}
	LDLT_INLINE auto segment(i32 i, i32 size) const noexcept
			-> StridedVectorViewMut {
		return {
				data + stride * i,
				size,
				stride,
		};
	}
};

// colmajor
template <typename Scalar, Layout L>
struct MatrixView {
	Scalar const* data;
	i32 rows;
	i32 cols;
	i32 outer_stride;

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept
			-> Scalar const& {
		return *detail::ElementAccess<L>::offset(data, row, col, outer_stride);
	}
	LDLT_INLINE auto block(i32 row, i32 col, i32 nrows, i32 ncols) const noexcept
			-> MatrixView {
		return {
				detail::ElementAccess<L>::offset(data, row, col, outer_stride),
				nrows,
				ncols,
				outer_stride,
		};
	}
	LDLT_INLINE auto col(i32 c) const noexcept -> VectorView<Scalar> {
		return {
				detail::ElementAccess<L>::offset(data, 0, c, outer_stride),
				rows,
		};
	}
	LDLT_INLINE auto row(i32 r) const noexcept -> StridedVectorView<Scalar> {
		return {
				detail::ElementAccess<L>::offset(data, r, 0, outer_stride),
				cols,
				outer_stride,
		};
	}
};
template <typename Scalar>
struct MatrixView<Scalar, rowmajor> {
	Scalar const* data;
	i32 rows;
	i32 cols;
	i32 outer_stride;

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept
			-> Scalar const& {
		return *detail::ElementAccess<rowmajor>::offset(
				data, row, col, outer_stride);
	}
	LDLT_INLINE auto block(i32 row, i32 col, i32 nrows, i32 ncols) const noexcept
			-> MatrixView {
		return {
				detail::ElementAccess<rowmajor>::offset(data, row, col, outer_stride),
				nrows,
				ncols,
				outer_stride,
		};
	}
	LDLT_INLINE auto row(i32 r) const noexcept -> VectorView<Scalar> {
		return {
				detail::ElementAccess<rowmajor>::offset(data, r, 0, outer_stride),
				cols,
		};
	}
	LDLT_INLINE auto col(i32 c) const noexcept -> StridedVectorView<Scalar> {
		return {
				detail::ElementAccess<rowmajor>::offset(data, 0, c, outer_stride),
				rows,
				outer_stride,
		};
	}
};

// colmajor
template <typename Scalar, Layout L>
struct MatrixViewMut {
	Scalar* data;
	i32 rows;
	i32 cols;
	i32 outer_stride;

	LDLT_INLINE auto as_const() const noexcept -> MatrixView<Scalar, L> {
		return {
				data,
				rows,
				cols,
				outer_stride,
		};
	}

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept -> Scalar& {
		return *detail::ElementAccess<L>::offset(data, row, col, outer_stride);
	}
	LDLT_INLINE auto block(i32 row, i32 col, i32 nrows, i32 ncols) const noexcept
			-> MatrixViewMut {
		return {
				detail::ElementAccess<L>::offset(data, row, col, outer_stride),
				nrows,
				ncols,
				outer_stride,
		};
	}
	LDLT_INLINE auto col(i32 c) const noexcept -> VectorViewMut<Scalar> {
		return {
				detail::ElementAccess<L>::offset(data, 0, c, outer_stride),
				rows,
		};
	}
	LDLT_INLINE auto row(i32 r) const noexcept -> StridedVectorViewMut<Scalar> {
		return {
				detail::ElementAccess<L>::offset(data, r, 0, outer_stride),
				cols,
				outer_stride,
		};
	}
};
template <typename Scalar>
struct MatrixViewMut<Scalar, rowmajor> {
	Scalar* data;
	i32 rows;
	i32 cols;
	i32 outer_stride;

	LDLT_INLINE auto as_const() const noexcept -> MatrixView<Scalar, rowmajor> {
		return {
				data,
				rows,
				cols,
				outer_stride,
		};
	}

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept -> Scalar& {
		return *detail::ElementAccess<rowmajor>::offset(
				data, row, col, outer_stride);
	}
	LDLT_INLINE auto block(i32 row, i32 col, i32 nrows, i32 ncols) const noexcept
			-> MatrixViewMut {
		return {
				detail::ElementAccess<rowmajor>::offset(data, row, col, outer_stride),
				nrows,
				ncols,
				outer_stride,
		};
	}
	LDLT_INLINE auto row(i32 r) const noexcept -> VectorViewMut<Scalar> {
		return {
				detail::ElementAccess<rowmajor>::offset(data, r, 0, outer_stride),
				cols,
		};
	}
	LDLT_INLINE auto col(i32 c) const noexcept -> StridedVectorViewMut<Scalar> {
		return {
				detail::ElementAccess<rowmajor>::offset(data, 0, c, outer_stride),
				rows,
				outer_stride,
		};
	}
};

template <typename Scalar, Layout L>
struct LdltView {
	MatrixView<Scalar, L> l;
	VectorView<Scalar> d;
};
template <typename Scalar, Layout L>
struct LdltViewMut {
	MatrixViewMut<Scalar, L> l;
	VectorViewMut<Scalar> d;

	LDLT_INLINE constexpr auto as_const() const noexcept -> LdltView<Scalar, L> {
		return {l.as_const(), d.as_const()};
	}
};

namespace detail {
template <typename T, Layout L>
using EigenMatMap = Eigen::Map<        //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				Eigen::Dynamic,                //
				(L == colmajor)                //
						? Eigen::ColMajor          //
						: Eigen::RowMajor          //
				> const,                       //
		Eigen::Unaligned,                  //
		Eigen::OuterStride<Eigen::Dynamic> //
		>;
template <typename T, Layout L>
using EigenMatMapMut = Eigen::Map<     //
		Eigen::Matrix<                     //
				T,                             //
				Eigen::Dynamic,                //
				Eigen::Dynamic,                //
				(L == colmajor)                //
						? Eigen::ColMajor          //
						: Eigen::RowMajor          //
				>,                             //
		Eigen::Unaligned,                  //
		Eigen::OuterStride<Eigen::Dynamic> //
		>;

template <typename T, typename Stride>
using EigenVecMap = Eigen::Map< //
		Eigen::Matrix<              //
				T,                      //
				Eigen::Dynamic,         //
				1                       //
				> const,                //
		Eigen::Unaligned,           //
		Stride                      //
		>;
template <typename T, typename Stride>
using EigenVecMapMut = Eigen::Map< //
		Eigen::Matrix<                 //
				T,                         //
				Eigen::Dynamic,            //
				1                          //
				>,                         //
		Eigen::Unaligned,              //
		Stride                         //
		>;

template <typename T, Layout L>
using ColToVec = EigenVecMap<T, typename ElementAccess<L>::NextRowStride>;
template <typename T, Layout L>
using RowToVec = EigenVecMap<T, typename ElementAccess<L>::NextColStride>;
template <typename T, Layout L>
using ColToVecMut = EigenVecMapMut<T, typename ElementAccess<L>::NextRowStride>;
template <typename T, Layout L>
using RowToVecMut = EigenVecMapMut<T, typename ElementAccess<L>::NextColStride>;

template <typename T>
using VecMap = EigenVecMap<T, Eigen::Stride<0, 0>>;
template <typename T>
using VecMapMut = EigenVecMapMut<T, Eigen::Stride<0, 0>>;

namespace nb {
struct from_eigen_matrix {
	template <typename T>
	LDLT_INLINE auto operator()(T const& mat) const noexcept -> MatrixView<
			typename T::Scalar,
			bool(T::IsRowMajor) //
					? rowmajor
					: colmajor> {
		return {
				(mat.data()),
				i32(mat.rows()),
				i32(mat.cols()),
				i32(mat.outerStride()),
		};
	}
};
struct from_eigen_matrix_mut {
	template <typename T>
	LDLT_INLINE auto operator()(T& mat) const noexcept -> MatrixViewMut<
			typename T::Scalar,
			bool(T::IsRowMajor) //
					? rowmajor
					: colmajor> {
		return {
				(mat.data()),
				i32(mat.rows()),
				i32(mat.cols()),
				i32(mat.outerStride()),
		};
	}
};
struct from_eigen_vector {
	template <typename T>
	LDLT_INLINE auto operator()(T const& vec) const noexcept
			-> VectorView<typename T::Scalar> {
		return {vec.data(), i32(vec.rows())};
	}
};

struct from_eigen_vector_mut {
	template <typename T>
	LDLT_INLINE auto operator()(T&& vec) const noexcept
			-> VectorViewMut<typename std::remove_reference<T>::type::Scalar> {
		return {vec.data(), i32(vec.rows())};
	}
};

struct to_eigen_matrix {
	template <typename T, Layout L>
	LDLT_INLINE auto operator()(MatrixView<T, L> mat) const noexcept
			-> EigenMatMap<T, L> {
		return {
				mat.data,
				mat.rows,
				mat.cols,
				mat.outer_stride,
		};
	}
};

struct to_eigen_matrix_mut {
	template <typename T, Layout L>
	LDLT_INLINE auto operator()(MatrixViewMut<T, L> mat) const noexcept
			-> EigenMatMapMut<T, L> {
		return {
				mat.data,
				mat.rows,
				mat.cols,
				mat.outer_stride,
		};
	}
};

struct to_eigen_vector {
	template <typename T>
	LDLT_INLINE auto operator()(VectorView<T> vec) const noexcept -> VecMap<T> {
		return {vec.data, vec.dim};
	}
	template <typename T>
	LDLT_INLINE auto operator()(StridedVectorView<T> vec) const noexcept
			-> EigenVecMap<T, Eigen::InnerStride<Eigen::Dynamic>> {
		return {
				vec.data,
				vec.dim,
				1,
				Eigen::InnerStride<Eigen::Dynamic>(vec.stride),
		};
	}
};

struct to_eigen_vector_mut {
	template <typename T>
	LDLT_INLINE auto operator()(VectorViewMut<T> vec) const noexcept
			-> VecMapMut<T> {
		return {vec.data, vec.dim};
	}
	template <typename T>
	LDLT_INLINE auto operator()(StridedVectorViewMut<T> vec) const noexcept
			-> EigenVecMapMut<T, Eigen::InnerStride<Eigen::Dynamic>> {
		return {
				vec.data,
				vec.dim,
				1,
				Eigen::InnerStride<Eigen::Dynamic>(vec.stride),
		};
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(from_eigen_matrix);
LDLT_DEFINE_NIEBLOID(from_eigen_matrix_mut);
LDLT_DEFINE_NIEBLOID(to_eigen_matrix);
LDLT_DEFINE_NIEBLOID(to_eigen_matrix_mut);

LDLT_DEFINE_NIEBLOID(from_eigen_vector);
LDLT_DEFINE_NIEBLOID(from_eigen_vector_mut);
LDLT_DEFINE_NIEBLOID(to_eigen_vector);
LDLT_DEFINE_NIEBLOID(to_eigen_vector_mut);
} // namespace detail
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_VIEWS_HPP_UGNXAQSBS */
