#ifndef INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS
#define INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS

#include "ldlt/views.hpp"
#include <new>

namespace ldlt {
namespace detail {
template <typename Scalar, Layout OutL, Layout InL>
LDLT_NO_INLINE void factorize_ldlt_tpl(
		LdltViewMut<Scalar, OutL> out, MatrixView<Scalar, InL> in_matrix) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2

	bool inplace = out.l.data == in_matrix.data;
	i32 dim = out.l.dim;
	i32 l_stride = out.l.outer_stride;

	LDLT_WORKSPACE_MEMORY(wp, dim, Scalar);

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
				detail::ElementAccess<OutL>::offset(out.l.data, 0, j, l_stride),
				j,
				1,
				detail::ElementAccess<OutL>::next_row_stride(l_stride),
		};
		l01.setZero();
		out.l(j, j) = Scalar(1);

		auto l10 = detail::RowToVec<Scalar, OutL>{
				detail::ElementAccess<OutL>::offset(out.l.data, j, 0, l_stride),
				j,
				1,
				detail::ElementAccess<OutL>::next_col_stride(l_stride),
		};

		auto l20 = EigenMatMap<Scalar, OutL>{
				detail::ElementAccess<OutL>::offset(out.l.data, j_inc, 0, l_stride),
				m,
				j,
				Eigen::OuterStride<Eigen::Dynamic>{l_stride},
		};
		auto l21 = detail::ColToVecMut<Scalar, OutL>{
				detail::ElementAccess<OutL>::offset(out.l.data, j_inc, j, l_stride),
				m,
				1,
				detail::ElementAccess<OutL>::next_row_stride(l_stride),
		};

		auto a21 = detail::ColToVec<Scalar, InL>{
				detail::ElementAccess<InL>::offset(
						in_matrix.data, j_inc, j, in_matrix.outer_stride),
				m,
				1,
				detail::ElementAccess<InL>::next_row_stride(in_matrix.outer_stride),
		};

		auto d = detail::VecMap<Scalar>{out.d.data, j};
		auto tmp_read = detail::VecMap<Scalar>{wp, j};
		auto tmp = detail::VecMapMut<Scalar>{wp, j};

		tmp.array() = l10.array().operator*(d.array());
		out.d(j) = in_matrix(j, j) - Scalar(tmp_read.dot(l10));
		if (!inplace) {
			l21 = a21;
		}
		l21.noalias().operator-=(l20.operator*(tmp_read));
		l21.operator*=(Scalar(1) / out.d(j));
	}
}
} // namespace detail

extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, colmajor>, MatrixView<f32, colmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, colmajor>, MatrixView<f64, colmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, rowmajor>, MatrixView<f32, colmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, rowmajor>, MatrixView<f64, colmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, colmajor>, MatrixView<f32, rowmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, colmajor>, MatrixView<f64, rowmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, colmajor>, MatrixView<f32, rowmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, colmajor>, MatrixView<f64, rowmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, rowmajor>, MatrixView<f32, rowmajor>);
extern template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, rowmajor>, MatrixView<f64, rowmajor>);

namespace nb {
struct factorize {
	template <typename Scalar, Layout OutL, Layout InL>
	LDLT_INLINE void operator()(
			LdltViewMut<Scalar, OutL> out, MatrixView<Scalar, InL> in_matrix) const {
		detail::factorize_ldlt_tpl(out, in_matrix);
	}
};

struct factorize_defer_to_colmajor {
	template <typename Scalar, Layout OutL, Layout InL>
	LDLT_INLINE void operator()(
			LdltViewMut<Scalar, OutL> out, MatrixView<Scalar, InL> in_matrix) const {
		detail::factorize_ldlt_tpl(
				LdltViewMut<Scalar, colmajor>{
						{out.l.data, out.l.dim, out.l.outer_stride},
						out.d,
				},
				in_matrix);
		detail::ElementAccess<OutL>::transpose_if_rowmajor( //
				out.l.data,
				out.l.dim,
				out.l.outer_stride);
	}
};
} // namespace nb

LDLT_DEFINE_NIEBLOID(factorize);
LDLT_DEFINE_NIEBLOID(factorize_defer_to_colmajor);

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS */
