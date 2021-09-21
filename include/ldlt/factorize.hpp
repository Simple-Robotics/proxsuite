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
template <typename T>
struct Indexed {
	T elem;
	i32 idx;
};
template <typename T>
LDLT_NO_INLINE void compute_permutation(
		i32* perm_indices, i32* perm_inv_indices, StridedVectorView<T> diagonal) {
	isize n = diagonal.dim;
	for (isize k = 0; k < n; ++k) {
		perm_indices[k] = i32(k);
	}

	{
		T const* diagonal_data = diagonal.data;
		isize stride = diagonal.stride;
		std::stable_sort(
				perm_indices,
				perm_indices + n,
				[diagonal_data, stride](i32 i, i32 j) noexcept -> bool {
					using std::fabs;
					return fabs(diagonal_data[stride * i]) >
			           fabs(diagonal_data[stride * j]);
				});
	}

	for (i32 k = 0; k < n; ++k) {
		i32 inv_k = perm_indices[k];
		perm_indices[k] = inv_k;
		perm_inv_indices[inv_k] = k;
	}
}
LDLT_EXPLICIT_TPL_DECL(3, compute_permutation<f32>);
LDLT_EXPLICIT_TPL_DECL(3, compute_permutation<f64>);

struct PermIterInfo {
	isize init;
	isize inc_start;
	isize inc_nrows;
};

inline auto perm_helper(i32 sym, isize nrows) -> PermIterInfo {
	switch (sym) {
	case -1:
		return {nrows, 1, -1};
	case 0:
		return {nrows, 0, 0};
	case 1:
		return {1, 0, 1};
	default:
		HEDLEY_UNREACHABLE();
	}
}

template <typename T>
struct apply_perm_rows {
	static void
	fn(T* out,
	   isize out_stride,
	   T* in,
	   isize in_stride,
	   isize ncols,
	   isize nrows,
	   i32 const* perm_indices,
	   i32 sym) noexcept {

		auto info = detail::perm_helper(sym, nrows);

		isize start_row = 0;
		isize current_nrows = info.init;

		isize const bytes_per_col = nrows * isize(sizeof(T));
		isize const n_prefetch =
				(bytes_per_col < 4096) ? 0 : (bytes_per_col / 64 * 64);

		for (isize col = 0; col < ncols; ++col) {
			T* in_c = in + (in_stride * col);
			T* out_c = out + (out_stride * col);

			if (std::is_trivially_copyable<T>::value) {
				for (isize i = 0; i < n_prefetch; i += 64) {
					simde_mm_prefetch(
							reinterpret_cast<char const*>(in_c) + i, SIMDE_MM_HINT_T0);
				}
			}

			for (isize row = start_row; row < start_row + current_nrows; ++row) {
				out_c[row] = static_cast<T&&>(in_c[perm_indices[row]]);
			}
			current_nrows += info.inc_nrows;
			start_row += info.inc_start;
		}
	}
};

// sym: -1 is lower half
//      +1 is upper half
//       0 is both
// when only one half is filled, the values in the other half are unspecified
template <typename T>
void apply_permutation_sym_work(
		MatrixViewMut<T, colmajor> mat,
		i32 const* perm_indices,
		MatrixViewMut<T, colmajor> work,
		i32 sym) {
	isize n = mat.rows;

	for (isize k = 0; k < n; ++k) {
		T* in = mat.col(isize(perm_indices[k])).data;
		T* out = work.col(k).data;
		std::move(in, in + n, out);
	}
	detail::apply_perm_rows<T>::fn(
			mat.data,
			mat.outer_stride,
			work.data,
			work.outer_stride,
			n,
			n,
			perm_indices,
			sym);
}

template <typename T>
LDLT_NO_INLINE void factorize_unblocked(LdltViewMut<T> out) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
	// TODO: use upper half of in_matrix?

	isize dim = out.l.rows;

	isize j = 0;
	while (true) {
		if (j == dim) {
			break;
		}
		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		auto l01_w = out.l.col(j).segment(0, j);
		auto l01 = out.l.col(j).segment(0, j).as_const();

		auto&& cleanup = defer([&] {
			// has to be after we're done using workspace because they may alias
			detail::set_zero(l01_w.data, usize(l01_w.dim));
			++j;
		});
		(void)cleanup;

		isize m = dim - j - 1;

		auto l_c = out.l.as_const();
		auto d_c = out.d.as_const();

		auto l10 = l_c.row(j).segment(0, j);
		auto d = d_c.segment(0, j);

		detail::assign_cwise_prod(l01_w, l10, d);
		out.d(j) = out.l(j, j) - detail::dot(l10, l01);
		out.l(j, j) = 1;

		if (j + 1 == dim) {
			break;
		}

		auto l20 = l_c.block(j + 1, 0, m, j);
		auto l21 = out.l.col(j).segment(j + 1, m);

		// l21 -= l20 * tmp_read
		detail::noalias_mul_add_vec(l21, l20, l01, T(-1));
		detail::assign_scalar_prod(l21, 1 / out.d(j), l21.as_const());
	}
}

template <typename T>
LDLT_NO_INLINE void factorize_blocked(LdltViewMut<T> ld, isize block_size) {

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

		detail::factorize_unblocked(LdltViewMut<T>{l11_mut, d1_mut});

		if (k + bs == n) {
			break;
		}

		isize rem = n - k - bs;
		auto l21_mut = ld.l.block(k + bs, k, rem, bs);
		auto l21 = l21_mut.as_const();
		detail::trans_tr_unit_up_solve_in_place_on_right(l11, l21_mut);
		detail::apply_diag_inv_on_right(l21_mut, d1, l21);

		auto work_k_mut = ld.l.block(0, n - bs, rem, bs);
		detail::apply_diag_on_right(work_k_mut, d1, l21);

		auto work_k = work_k_mut.as_const();

		auto l22_mut = ld.l.block(k + bs, k + bs, rem, rem);
		detail::noalias_mul_sub_tr_lo(l22_mut, l21, work_k.trans());
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
		isize dim = out.l.rows;
		bool inplace = (out.l.data == in_matrix.data) && (L == colmajor);
		if (!inplace) {
			out.l.to_eigen().template triangularView<Eigen::Lower>() =
					in_matrix.to_eigen();
		}
		detail::factorize_unblocked(out);
		for (isize k = 0; k < dim; ++k) {
			detail::set_zero(out.l.col(k).data, usize(k));
		}
	}
};
template <>
struct FactorizeStartegyDispatch<factorization_strategy::Blocked> {
	template <typename T, Layout L>
	static LDLT_INLINE void
	fn(LdltViewMut<T> out,
	   MatrixView<T, L> in_matrix,
	   factorization_strategy::Blocked tag) {
		isize dim = out.l.rows;
		bool inplace = (out.l.data == in_matrix.data) && (L == colmajor);
		if (!inplace) {
			out.l.to_eigen().template triangularView<Eigen::Lower>() =
					in_matrix.to_eigen();
		}
		detail::factorize_blocked(out, tag.block_size);
		for (isize k = 0; k < dim; ++k) {
			detail::set_zero(out.l.col(k).data, usize(k));
		}
	}
};

LDLT_EXPLICIT_TPL_DECL(1, factorize_unblocked<f32>);
LDLT_EXPLICIT_TPL_DECL(2, factorize_blocked<f32>);
LDLT_EXPLICIT_TPL_DECL(1, factorize_unblocked<f64>);
LDLT_EXPLICIT_TPL_DECL(2, factorize_blocked<f64>);

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
