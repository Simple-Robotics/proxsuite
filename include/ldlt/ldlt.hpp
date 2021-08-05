#ifndef LDLT_LDLT_HPP_FDFNWYGES
#define LDLT_LDLT_HPP_FDFNWYGES

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include <memory>
#include <vector>
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

	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept
			-> Scalar const& {
		return *detail::ElementAccess<L>::offset(data, row, col, dim);
	}
};

template <typename Scalar, Layout L>
struct MatrixViewMut {
	Scalar* data;
	i32 dim;

	LDLT_INLINE auto as_const() const noexcept -> MatrixView<Scalar, L> {
		return {data, dim};
	}
	LDLT_INLINE auto operator()(i32 row, i32 col) const noexcept -> Scalar& {
		return *detail::ElementAccess<L>::offset(data, row, col, dim);
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

#ifdef __clang__
#define LDLT_FP_PRAGMA _Pragma("STDC FP_CONTRACT ON")
#else
#define LDLT_FP_PRAGMA
#endif

namespace detail {
template <Layout L>
struct MatrixLoadRowBlock;

template <>
struct MatrixLoadRowBlock<Layout::rowmajor> {
	template <usize N, typename Scalar>
	LDLT_INLINE static auto load_pack(Scalar const* p, i32 stride) noexcept
			-> Pack<Scalar, N> {
		(void)stride;
		return Pack<Scalar, N>::load_unaligned(p);
	}
};
template <>
struct MatrixLoadRowBlock<Layout::colmajor> {
	template <usize N, typename Scalar>
	LDLT_INLINE static auto load_pack(Scalar const* p, i32 stride) noexcept
			-> Pack<Scalar, N> {
		return Pack<Scalar, N>::load_gather(p, stride);
	}
};

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

namespace detail {
template <typename Scalar, Layout OutL, Layout InL>
LDLT_NO_INLINE void factorize_ldlt_tpl(
		MatrixViewMut<Scalar, OutL> out_l,
		DiagonalMatrixViewMut<Scalar> out_d,
		MatrixView<Scalar, InL> in_matrix) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2

	i32 dim = out_l.dim;

	auto workspace = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(dim);

	for (i32 j = 0; j < dim; ++j) {
		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		i32 m = dim - j - 1;

		// avoid buffer overflow UB when accessing the matrices
		i32 j_inc = ((j + 1) < dim) ? j + 1 : j;

		auto l01 = detail::ColToVecMut<Scalar, OutL>{
				detail::ElementAccess<OutL>::offset(out_l.data, 0, j, dim),
				j,
				1,
				detail::ElementAccess<OutL>::next_row_stride(dim),
		};
		l01.setZero();
		out_l(j, j) = Scalar(1);

		auto l10 = detail::RowToVec<Scalar, OutL>{
				detail::ElementAccess<OutL>::offset(out_l.data, j, 0, dim),
				j,
				1,
				detail::ElementAccess<OutL>::next_col_stride(dim),
		};

		auto l20 = Eigen::Map<                 //
				Eigen::Matrix<                     //
						Scalar,                        //
						Eigen::Dynamic,                //
						Eigen::Dynamic,                //
						(OutL == Layout::colmajor)     //
								? Eigen::ColMajor          //
								: Eigen::RowMajor          //
						> const,                       //
				Eigen::Unaligned,                  //
				Eigen::OuterStride<Eigen::Dynamic> //
				>{
				detail::ElementAccess<OutL>::offset(out_l.data, j_inc, 0, dim),
				m,
				j,
				Eigen::OuterStride<Eigen::Dynamic>{dim},
		};
		auto l21 = detail::ColToVecMut<Scalar, OutL>{
				detail::ElementAccess<OutL>::offset(out_l.data, j_inc, j, dim),
				m,
				1,
				detail::ElementAccess<OutL>::next_row_stride(dim),
		};

		auto a21 = detail::ColToVec<Scalar, InL>{
				detail::ElementAccess<InL>::offset(in_matrix.data, j_inc, j, dim),
				m,
				1,
				detail::ElementAccess<InL>::next_row_stride(dim),
		};

		auto d = detail::VecMap<Scalar>{out_d.data, j};
		auto tmp_read = detail::VecMap<Scalar>{workspace.data(), j};
		auto tmp = detail::VecMapMut<Scalar>{workspace.data(), j};

		tmp.array() = l10.array().operator*(d.array());
		out_d(j) = in_matrix(j, j) - Scalar(tmp_read.dot(l10));
		l21 = a21;
		l21.noalias().operator-=(l20.operator*(tmp_read));
		l21 = l21.operator/(out_d(j));
	}
}
} // namespace detail

namespace nb {
struct factorize {
	template <typename Scalar, Layout OutL, Layout InL>
	LDLT_INLINE void operator()(
			MatrixViewMut<Scalar, OutL> out_l,
			DiagonalMatrixViewMut<Scalar> out_d,
			MatrixView<Scalar, InL> in_matrix) const {
		detail::factorize_ldlt_tpl(out_l, out_d, in_matrix);
	}
};

struct factorize_defer_to_colmajor {
	template <typename Scalar, Layout OutL, Layout InL>
	LDLT_INLINE void operator()(
			MatrixViewMut<Scalar, OutL> out_l,
			DiagonalMatrixViewMut<Scalar> out_d,
			MatrixView<Scalar, InL> in_matrix) const {
		detail::factorize_ldlt_tpl(
				MatrixViewMut<Scalar, colmajor>{
						out_l.data,
						out_l.dim,
				},
				out_d,
				in_matrix);
		detail::ElementAccess<OutL>::transpose_if_rowmajor(
				out_l.data, out_l.dim, out_l.dim);
	}
};
} // namespace nb

LDLT_DEFINE_NIEBLOID(factorize);
LDLT_DEFINE_NIEBLOID(factorize_defer_to_colmajor);

} // namespace ldlt

#endif /* end of include guard LDLT_LDLT_HPP_FDFNWYGES */
