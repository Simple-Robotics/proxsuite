#ifndef LDLT_LDLT_HPP_FDFNWYGES
#define LDLT_LDLT_HPP_FDFNWYGES

#include "ldlt/views.hpp"

namespace ldlt {

namespace detail {
template <typename Scalar, Layout OutL, Layout InL>
LDLT_NO_INLINE void factorize_ldlt_tpl(
		MatrixViewMut<Scalar, OutL> out_l,
		DiagonalMatrixViewMut<Scalar> out_d,
		MatrixView<Scalar, InL> in_matrix) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2

	bool inplace = out_l.data == in_matrix.data;
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
		if (!inplace) {
			l21 = a21;
		}
		l21.noalias().operator-=(l20.operator*(tmp_read));
		l21.operator*=(Scalar(1) / out_d(j));
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
