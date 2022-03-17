#ifndef INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS
#define INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS

#include "ldlt/views.hpp"
#include <algorithm>

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
	isize idx;
};
template <typename T>
LDLT_NO_INLINE void compute_permutation(
		isize* perm_indices,
		isize* perm_inv_indices,
		StridedVectorView<T> diagonal) {
	isize n = diagonal.dim;
	for (isize k = 0; k < n; ++k) {
		perm_indices[k] = k;
	}

	{
		T const* diagonal_data = diagonal.data;
		isize stride = diagonal.stride;
		std::sort(
				perm_indices,
				perm_indices + n,
				[diagonal_data, stride](isize i, isize j) noexcept -> bool {
					using std::fabs;
					auto lhs = fabs(diagonal_data[stride * i]);
					auto rhs = fabs(diagonal_data[stride * j]);
					if (lhs == rhs) {
						return i < j;
					}
					return lhs > rhs;
				});
	}

	for (isize k = 0; k < n; ++k) {
		perm_inv_indices[perm_indices[k]] = k;
	}
}

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
	   isize nrows,
	   isize ncols,
	   isize const* perm_indices,
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
		isize const* perm_indices,
		MatrixViewMut<T, colmajor> work,
		i32 sym) {
	isize n = mat.rows;
	VEG_ASSERT_ALL_OF( //
			n == mat.rows,
			n == mat.cols,
			n == work.rows,
			n == work.cols);

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
VEG_NO_INLINE void factorize_unblocked(LdltViewMut<T> out) {
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
	// TODO: use upper half of in_matrix?

	isize dim = out.d().dim;
	if (dim == 0) {
		return;
	}

	isize i = 0;
	while (true) {
		auto&& _ = veg::defer([&i] { ++i; });
		veg::unused(_);

		if (i == dim) {
			break;
		}

		/********************************************************************************
		 *     l00 l01 l02
		 * l = l10  1  l12
		 *     l20 l21 l22
		 *
		 * l{0,1,2}0 already known, compute l21
		 */

		auto l01_mut = out.l_mut().col(i).segment(0, i);
		auto l01 = out.l().col(i).segment(0, i);

		isize m = dim - i - 1;

		auto l10 = out.l().row(i).segment(0, i);
		auto d0 = out.d().segment(0, i);

		detail::assign_cwise_prod(l01_mut, l10, d0);
		out.d_mut()(i) -= detail::dot(l10, l01);

		if (i + 1 == dim) {
			break;
		}

		auto l20 = out.l().block(i + 1, 0, m, i);
		auto l21_mut = out.l_mut().col(i).segment(i + 1, m);

		// l21 -= l20 * tmp_read
		detail::noalias_mul_add_vec(l21_mut, l20, l01, T(-1));
		detail::assign_scalar_prod(l21_mut, 1 / out.d()(i), l21_mut.as_const());
	}
}

template <typename T>
VEG_NO_INLINE void factorize_blocked(LdltViewMut<T> ld, isize block_size) {

	isize n = ld.d().dim;
	if (n <= 0) {
		return;
	}

	isize i = 0;
	while (true) {
		isize bs = min2(n - i, block_size);
		auto&& _ = veg::defer([&] { i += bs; });
		veg::unused(_);

		auto l11_mut = ld.l_mut().block(i, i, bs, bs);
		auto d1_mut = ld.d_mut().segment(i, bs);

		auto l11 = l11_mut.as_const();
		auto d1 = d1_mut.as_const();

		detail::factorize_unblocked(LdltViewMut<T>{{l11_mut}});

		if (i + bs == n) {
			break;
		}

		isize rem = n - i - bs;
		auto l21_mut = ld.l_mut().block(i + bs, i, rem, bs);
		auto l21 = l21_mut.as_const();
		detail::trans_tr_unit_up_solve_in_place_on_right(l11, l21_mut);
		detail::apply_diag_inv_on_right(l21_mut, d1, l21);

		auto work_k_mut = ld.l_mut().block(0, n - bs, rem, bs);
		detail::apply_diag_on_right(work_k_mut, d1, l21);

		auto work_k = work_k_mut.as_const();

		auto l22_mut = ld.l_mut().block(i + bs, i + bs, rem, rem);
		detail::noalias_mul_sub_tr_lo(l22_mut, l21, work_k.trans());
	}
}

template <typename S>
struct FactorizeStartegyDispatch;

template <>
struct FactorizeStartegyDispatch<factorization_strategy::Standard> {
	template <typename T>
	static LDLT_INLINE void
	fn(LdltViewMut<T> inout, factorization_strategy::Standard /*tag*/) {
		detail::factorize_unblocked(inout);
	}
};
template <>
struct FactorizeStartegyDispatch<factorization_strategy::Blocked> {
	template <typename T>
	static LDLT_INLINE void
	fn(LdltViewMut<T> inout, factorization_strategy::Blocked tag) {
		detail::factorize_blocked(inout, tag.block_size);
	}
};

} // namespace detail

namespace nb {
struct factorize {
	template <typename T, typename Strategy = factorization_strategy::Standard>
	LDLT_INLINE void
	operator()(LdltViewMut<T> inout, Strategy tag = Strategy{}) const {
		detail::FactorizeStartegyDispatch<Strategy>::fn(inout, tag);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(factorize);

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_FACTORIZE_HPP_FOK6CBQFS */
