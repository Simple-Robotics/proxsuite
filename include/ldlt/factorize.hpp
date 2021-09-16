#ifndef INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS
#define INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS

#include "ldlt/views.hpp"
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <array>

namespace ldlt {
namespace factorization_strategy {
LDLT_DEFINE_TAG(standard, Standard);
LDLT_DEFINE_TAG(experimental, Experimental);
inline namespace tags {
struct Blocked {
	isize block_size;
};
struct AltBlocked {
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
namespace nb {
struct alt_blocked {
	auto operator()(isize block_size) const noexcept -> AltBlocked {
		return {block_size};
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(blocked);
LDLT_DEFINE_NIEBLOID(alt_blocked);
} // namespace factorization_strategy

namespace detail {
template <typename T>
struct Indexed {
	T elem;
	i32 idx;
};
template <typename T>
LDLT_NO_INLINE void compute_permutation(
		i32* perm_indices, i32* perm_inv_indices, StridedVectorView<T> diagonal) {
	isize n = diagonal.dim;
	LDLT_WORKSPACE_MEMORY(work, Vec(n), Indexed<T>);
	for (isize k = 0; k < n; ++k) {
		using std::fabs;
		work(k) = {fabs(diagonal(k)), i32(k)};
	}
	std::stable_sort(
			work.data,
			work.data + n,
			[](Indexed<T> a, Indexed<T> b) noexcept -> bool {
				return a.elem > b.elem;
			});

	for (i32 k = 0; k < n; ++k) {
		i32 inv_k = work(k).idx;
		perm_indices[k] = inv_k;
		perm_inv_indices[inv_k] = k;
	}
}

template <typename T>
struct apply_perm_rows {
	static void
	fn(T* out,
	   isize out_stride,
	   T const* in,
	   isize in_stride,
	   isize n,
	   i32 const* perm_indices) noexcept {
		for (isize row = 0; row < n; ++row) {
			i32 indices0 = perm_indices[row];
			for (isize col = 0; col < n; ++col) {
				out[out_stride * col + row] = in[in_stride * col + indices0];
			}
		}
	}
};

template <>
struct apply_perm_rows<f64> {
	static void
	fn(f64* out,
	   isize out_stride,
	   f64 const* in,
	   isize in_stride,
	   isize n,
	   i32 const* perm_indices) noexcept;
};

template <>
struct apply_perm_rows<f32> {
	static void
	fn(f32* out,
	   isize out_stride,
	   f32 const* in,
	   isize in_stride,
	   isize n,
	   i32 const* perm_indices) noexcept;
};

template <typename T>
void apply_permutation_sym_work(
		MatrixViewMut<T, colmajor> mat,
		i32 const* perm_indices,
		MatrixViewMut<T, colmajor> work) {
	isize n = mat.rows;

	detail::apply_perm_rows<T>::fn(
			work.data,
			work.outer_stride,
			mat.data,
			mat.outer_stride,
			n,
			perm_indices);

	for (isize k = 0; k < n; ++k) {
		// already vectorized
		mat.col(k).to_eigen() =
				work.col(isize(perm_indices[k])).as_const().to_eigen();
	}
}

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

template <typename T>
LDLT_NO_INLINE void
factorize_ldlt_alt_inplace(LdltViewMut<T> ld, isize block_size) {

	isize n = ld.l.rows;
	if (n <= 0) {
		return;
	}

	isize k = 0;
	while (true) {
		isize bs = min2(n - k, block_size);

		auto l11_mut = ld.l.block(k, k, bs, bs);
		auto d1_mut = ld.d.segment(k, bs);

		auto l11 = l11_mut.as_const();
		auto d1 = d1_mut.as_const();

		detail::factorize_ldlt_tpl(
				LdltViewMut<T>{l11_mut, d1_mut}, l11_mut.as_const(), 1);

		if (k + bs == n) {
			break;
		}

		isize rem = n - k - bs;
		auto l21_mut = ld.l.block(k + bs, k, rem, bs);
		auto l21 = l21_mut.as_const();
		auto l22_mut = ld.l.block(k + bs, k + bs, rem, rem);
		l11.to_eigen()
				.transpose()
				.template triangularView<Eigen::UnitUpper>()
				.template solveInPlace<Eigen::OnTheRight>(l21_mut.to_eigen());
		l21_mut.to_eigen() = l21.to_eigen() * d1.to_eigen().asDiagonal().inverse();

		auto work_k_mut = ld.l.block(0, n - bs, rem, bs);
		auto work_k = work_k_mut.as_const();
		work_k_mut.to_eigen() = l21.to_eigen() * d1.to_eigen().asDiagonal();
		l22_mut.to_eigen().template triangularView<Eigen::Lower>() -=
				l21.to_eigen() * work_k.trans().to_eigen();

		k += bs;
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
struct FactorizeStartegyDispatch<factorization_strategy::AltBlocked> {
	template <typename T, Layout L>
	static LDLT_INLINE void
	fn(LdltViewMut<T> out,
	   MatrixView<T, L> in_matrix,
	   factorization_strategy::AltBlocked tag) {
		isize dim = out.l.rows;
		out.l.to_eigen().template triangularView<Eigen::Lower>() =
				in_matrix.to_eigen();
		detail::factorize_ldlt_alt_inplace(out, tag.block_size);
		for (isize k = 0; k < dim; ++k) {
			detail::set_zero(out.l.col(k).data, usize(k));
		}
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
