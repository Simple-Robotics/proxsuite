#ifndef INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS
#define INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS

#include "ldlt/views.hpp"

namespace ldlt {
namespace factorization_strategy {
LDLT_DEFINE_TAG(standard, Standard);
LDLT_DEFINE_TAG(experimental, Experimental);
} // namespace factorization_strategy

namespace detail {
template <typename T, Layout L>
LDLT_NO_INLINE void
factorize_ldlt_tpl(LdltViewMut<T> out, MatrixView<T, L> in_matrix) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
	// TODO: use upper half of in_matrix

	bool inplace = (out.l.data == in_matrix.data);
	isize dim = out.l.rows;

	LDLT_WORKSPACE_MEMORY(wp, Vec(dim), T);

	for (isize j = 0; j < dim; ++j) {
		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		isize m = dim - j - 1;

		// avoid buffer overflow UB when accessing the matrices
		isize j_inc = ((j + 1) < dim) ? j + 1 : j;

		auto l01 = out.l.col(j).segment(0, j).to_eigen();
		l01.setZero();
		T in_diag = in_matrix(j, j);
		out.l(j, j) = T(1);

		auto l_c = out.l.as_const();
		auto d_c = out.d.as_const();

		auto l10 = l_c.row(j).segment(0, j).to_eigen();
		auto l20 = l_c.block(j_inc, 0, m, j).to_eigen();
		auto l21 = out.l.col(j).segment(j_inc, m).to_eigen();
		auto a21 = in_matrix.col(j).segment(j_inc, m).to_eigen();
		auto d = d_c.segment(0, j).to_eigen();

		{
			auto tmp = wp.segment(0, j).to_eigen();
			tmp.array() = l10.array().operator*(d.array());
		}
		auto tmp_read = wp.as_const().segment(0, j).to_eigen();

		out.d(j) = in_diag - T(tmp_read.dot(l10));
		if (!inplace) {
			l21 = a21;
		}
		l21.noalias().operator-=(l20.operator*(tmp_read));
		l21.operator*=(T(1) / out.d(j));
	}
}

template <typename S>
struct FactorizeStartegyDispatch;

template <>
struct FactorizeStartegyDispatch<factorization_strategy::Standard> {
	template <typename T, Layout L>
	static LDLT_INLINE void fn(LdltViewMut<T> out, MatrixView<T, L> in_matrix) {
		detail::factorize_ldlt_tpl(out, in_matrix);
	}
};
template <>
struct FactorizeStartegyDispatch<factorization_strategy::Experimental> {
	template <typename T, Layout L>
	static LDLT_INLINE void fn(LdltViewMut<T> out, MatrixView<T, L> in_matrix) {
		// TODO: use faster matrix-vector product?
		detail::factorize_ldlt_tpl(out, in_matrix);
	}
};

extern template void
		factorize_ldlt_tpl(LdltViewMut<f32>, MatrixView<f32, colmajor>);
extern template void
		factorize_ldlt_tpl(LdltViewMut<f64>, MatrixView<f64, colmajor>);
extern template void
		factorize_ldlt_tpl(LdltViewMut<f32>, MatrixView<f32, rowmajor>);
extern template void
		factorize_ldlt_tpl(LdltViewMut<f64>, MatrixView<f64, rowmajor>);

} // namespace detail

namespace nb {
struct factorize {
	template <
			typename T,
			Layout L,
			typename Strategy = factorization_strategy::Standard>
	LDLT_INLINE void operator()(
			LdltViewMut<T> out,
			MatrixView<T, L> in_matrix,
			Strategy /*tag*/ = Strategy{}) const {
		detail::FactorizeStartegyDispatch<Strategy>::fn(out, in_matrix);
	}
};
} // namespace nb

LDLT_DEFINE_NIEBLOID(factorize);

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS */
