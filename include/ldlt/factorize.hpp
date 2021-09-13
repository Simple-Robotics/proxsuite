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
LDLT_NO_INLINE void factorize_ldlt_tpl(
		LdltViewMut<T> out, MatrixView<T, L> in_matrix, isize block_size) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
	// TODO: use upper half of in_matrix?

	bool inplace = (out.l.data == in_matrix.data);
	isize dim = out.l.rows;

	isize unblocked_n = (block_size == 1) ? dim : (dim % block_size);

	isize j = 0;
	while (true) {
		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		auto l01 = out.l.col(j).segment(0, j).to_eigen();

		auto&& cleanup = defer([&] {
			// has to be after we're done using workspace because they may alias
			detail::set_zero(l01.data(), usize(l01.size()));
			++j;
		});
		(void)cleanup;

		isize m = dim - j - 1;

		T in_diag = in_matrix(j, j);
		out.l(j, j) = T(1);

		auto l_c = out.l.as_const();
		auto d_c = out.d.as_const();

		auto l10 = l_c.row(j).segment(0, j).to_eigen();
		auto d = d_c.segment(0, j).to_eigen();

		auto wp_j =
				out.l.col(dim - 1).segment(0, j); // contiguous cause l is column major

		wp_j.to_eigen() = l10.cwiseProduct(d);
		auto tmp_read = wp_j.as_const().to_eigen();

		out.d(j) = in_diag - T(tmp_read.dot(l10));

		if (j + 1 == unblocked_n) {
			break;
		}

		auto l20 = l_c.block(j + 1, 0, m, j).to_eigen();
		auto l21 = out.l.col(j).segment(j + 1, m).to_eigen();
		auto a21 = in_matrix.col(j).segment(j + 1, m).to_eigen();

		if (!inplace) {
			l21 = a21;
		}
		l21.noalias() -= l20 * tmp_read;
		l21 *= T(1) / out.d(j);
	}

	if (j == dim) {
		return;
	}

	while (true) {
		auto&& cleanup = defer([&] { ++j; });
		(void)cleanup;
	}
}

template <typename S>
struct FactorizeStartegyDispatch;

template <>
struct FactorizeStartegyDispatch<factorization_strategy::Standard> {
	template <typename T, Layout L>
	static LDLT_INLINE void fn(LdltViewMut<T> out, MatrixView<T, L> in_matrix) {
		detail::factorize_ldlt_tpl(out, in_matrix, 1);
	}
};
template <>
struct FactorizeStartegyDispatch<factorization_strategy::Experimental> {
	template <typename T, Layout L>
	static LDLT_INLINE void fn(LdltViewMut<T> out, MatrixView<T, L> in_matrix) {
		// TODO: use faster matrix-vector product?
		detail::factorize_ldlt_tpl(out, in_matrix, -1);
	}
};

extern template void
		factorize_ldlt_tpl(LdltViewMut<f32>, MatrixView<f32, colmajor>, isize);
extern template void
		factorize_ldlt_tpl(LdltViewMut<f64>, MatrixView<f64, colmajor>, isize);
extern template void
		factorize_ldlt_tpl(LdltViewMut<f32>, MatrixView<f32, rowmajor>, isize);
extern template void
		factorize_ldlt_tpl(LdltViewMut<f64>, MatrixView<f64, rowmajor>, isize);

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
