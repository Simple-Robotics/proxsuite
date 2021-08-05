#ifndef INRIA_LDLT_VIEWS_HPP_UGNXAQSBS
#define INRIA_LDLT_VIEWS_HPP_UGNXAQSBS

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include <Eigen/Core>

namespace ldlt {
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

template <typename Scalar, Layout L>
struct MatrixView {
	Scalar const* data;
	i32 dim;
	i32 outer_stride;

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept
			-> Scalar const& {
		return *detail::ElementAccess<L>::offset(data, row, col, outer_stride);
	}
};

template <typename Scalar, Layout L>
struct MatrixViewMut {
	Scalar* data;
	i32 dim;
	i32 outer_stride;

	LDLT_INLINE auto as_const() const noexcept -> MatrixView<Scalar, L> {
		return {data, dim, outer_stride};
	}
	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept -> Scalar& {
		return *detail::ElementAccess<L>::offset(data, row, col, outer_stride);
	}
};

template <typename Scalar>
struct DiagonalMatrixView {
	Scalar const* data;
	i32 dim;

	HEDLEY_ALWAYS_INLINE auto operator()(i32 index) const noexcept
			-> Scalar const& {
		return *(data + index);
	}
};

template <typename Scalar>
struct DiagonalMatrixViewMut {
	Scalar* data;
	i32 dim;
	LDLT_INLINE auto as_const() const noexcept -> DiagonalMatrixView<Scalar> {
		return {data, dim};
	}
	HEDLEY_ALWAYS_INLINE auto operator()(i32 index) const noexcept -> Scalar& {
		return *(data + index);
	}
};

namespace detail {
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
} // namespace detail
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_VIEWS_HPP_UGNXAQSBS */
