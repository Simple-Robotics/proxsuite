#ifndef INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS
#define INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS

#include "ldlt/views.hpp"

namespace ldlt {
namespace factorization_strategy {
LDLT_DEFINE_TAG(standard, Standard);
LDLT_DEFINE_TAG(experimental, Experimental);
inline namespace tags {
struct Blocked {
	isize block_size;
};
} // namespace tags
namespace nb {
struct blocked {
	auto operator()(isize block_size) const noexcept -> Blocked {
		return {block_size};
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(blocked);
} // namespace factorization_strategy

namespace detail {
template <typename T, Layout L>
LDLT_NO_INLINE void
factorize_ldlt_tpl(LdltViewMut<T> out, MatrixView<T, L> in_matrix, isize bs) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
	// TODO: use upper half of in_matrix?

	bool inplace = (out.l.data == in_matrix.data);
	isize dim = out.l.rows;

	isize unblocked_n = (bs == 1) ? dim : min2(dim, bs + dim % bs);

	isize j = 0;
	while (true) {
		if (j == unblocked_n) {
			break;
		}
		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		auto l01_w = out.l.col(j).segment(0, j).to_eigen();
		auto l01 = out.l.col(j).segment(0, j).as_const().to_eigen();

		auto&& cleanup = defer([&] {
			// has to be after we're done using workspace because they may alias
			detail::set_zero(l01_w.data(), usize(l01_w.size()));
			++j;
		});
		(void)cleanup;

		isize m = dim - j - 1;

		auto l_c = out.l.as_const();
		auto d_c = out.d.as_const();

		auto l10 = l_c.row(j).segment(0, j).to_eigen();
		auto d = d_c.segment(0, j).to_eigen();

		l01_w = l10.cwiseProduct(d);

		out.d(j) = in_matrix(j, j) - l01.dot(l10);
		out.l(j, j) = 1;

		if (j + 1 == dim) {
			break;
		}

		auto l20 = l_c.block(j + 1, 0, m, j);
		auto l21 = out.l.col(j).segment(j + 1, m);
		auto a21 = in_matrix.col(j).segment(j + 1, m);

		if (!inplace) {
			l21.to_eigen() = a21.to_eigen();
		}

		// l21 -= l20 * tmp_read
		detail::noalias_mul_add_vec<T>( //
				l21,
				l20,
				{from_eigen, l01},
				T(-1));

		l21.to_eigen() *= 1 / out.d(j);
	}

	if (j == dim) {
		return;
	}

	while (true) {
		auto&& cleanup = defer([&] {
			for (isize k = j; k < j + bs; ++k) {
				detail::set_zero(out.l.ptr(0, k), usize(k));
			}
			j += bs;
		});
		(void)cleanup;

		for (isize k = j; k < j + bs; ++k) {
			auto l01_w = out.l.col(k).segment(0, k).to_eigen();
			auto l01 = out.l.col(k).segment(0, k).as_const().to_eigen();
			auto l10 = out.l.row(k).segment(0, k).as_const().to_eigen();

			auto d = out.d.segment(0, k).to_eigen();

			// TODO: optimize by simdifying over k?
			l01_w = l10.cwiseProduct(d);

			out.d(k) = in_matrix(k, k) - l01.dot(l10);
			out.l(k, k) = 1;
			if ((k + 1) == dim) {
				return;
			}

			isize nrows = bs - 1 - (k - j);

			auto l21 = out.l.col(k).segment(k + 1, nrows);
			auto l20 = out.l.block(k + 1, 0, nrows, k).as_const();
			if (!inplace) {
				l21.to_eigen() = in_matrix.col(k).segment(k + 1, nrows).to_eigen();
			}

			detail::noalias_mul_add_vec<T>( //
					l21,
					l20,
					{from_eigen, l01},
					T(-1));

			l21.to_eigen() *= 1 / out.d(k);
		}

		isize m = dim - j - bs;

		auto l01 = out.l.block(0, j, j, bs).as_const();
		auto l20 = out.l.block(j + bs, 0, m, j).as_const();
		auto l21 = out.l.block(j + bs, j, m, bs);
		auto a21 = in_matrix.block(j + bs, j, m, bs);

		if (!inplace) {
			l21.to_eigen() = a21.to_eigen();
		}
		detail::noalias_mul_add<T>(l21, l20, l01, T(-1));

		for (isize p = 0; p < bs; ++p) {
			auto block = out.l.block(j, j, bs, bs).as_const();

			detail::noalias_mul_add_vec<T>( //
					l21.col(p),
					l21.block(0, 0, m, p).as_const(),
					block.col(p).segment(0, p),
					T(-1));

			l21.col(p).to_eigen() *= 1 / out.d(j + p);
		}
	}
}

template <typename S>
struct FactorizeStartegyDispatch;

template <>
struct FactorizeStartegyDispatch<factorization_strategy::Standard> {
	template <typename T, Layout L>
	static LDLT_INLINE void
	fn(LdltViewMut<T> out,
	   MatrixView<T, L> in_matrix,
	   factorization_strategy::Standard /*tag*/) {
		detail::factorize_ldlt_tpl(out, in_matrix, 1);
	}
};
template <>
struct FactorizeStartegyDispatch<factorization_strategy::Blocked> {
	template <typename T, Layout L>
	static LDLT_INLINE void
	fn(LdltViewMut<T> out,
	   MatrixView<T, L> in_matrix,
	   factorization_strategy::Blocked tag) {
		detail::factorize_ldlt_tpl(out, in_matrix, tag.block_size);
	}
};
template <>
struct FactorizeStartegyDispatch<factorization_strategy::Experimental> {
	template <typename T, Layout L>
	static LDLT_INLINE void
	fn(LdltViewMut<T> out,
	   MatrixView<T, L> in_matrix,
	   factorization_strategy::Experimental /*tag*/) {
		// TODO: use faster matrix-vector product?
		detail::factorize_ldlt_tpl(out, in_matrix, 1);
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
			Strategy tag = Strategy{}) const {
		detail::FactorizeStartegyDispatch<Strategy>::fn(out, in_matrix, tag);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(factorize);

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS */
