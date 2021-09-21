#include <ldlt/factorize.hpp>

namespace ldlt {
namespace detail {
LDLT_EXPLICIT_TPL_DEF(1, factorize_unblocked<f32>);
LDLT_EXPLICIT_TPL_DEF(2, factorize_blocked<f32>);
LDLT_EXPLICIT_TPL_DEF(1, factorize_unblocked<f64>);
LDLT_EXPLICIT_TPL_DEF(2, factorize_blocked<f64>);
LDLT_EXPLICIT_TPL_DEF(3, compute_permutation<f32>);
LDLT_EXPLICIT_TPL_DEF(3, compute_permutation<f64>);

#ifdef __clang__
#define LDLT_MINSIZE __attribute__((minsize))
#else
#define LDLT_MINSIZE
#endif

LDLT_MINSIZE
void apply_perm_rows<f64>::fn(
		f64* out,
		isize out_stride,
		f64 const* in,
		isize in_stride,
		isize n,
		i32 const* perm_indices,
		i32 sym) noexcept {

	auto info = detail::perm_helper(sym, n);

	isize n8 = n / 8 * 8;

	isize n_prefetch = n < 512 ? 0 : n8;

	isize start_row = 0;
	isize nrows = info.init;

	for (isize col = 0; col < n; ++col) {
		f64* out_c = out + out_stride * col;
		f64 const* in_c = in + in_stride * col;

		isize last_row = start_row + nrows;

		for (isize i = 0; i < n_prefetch; i += 8) {
			simde_mm_prefetch(in_c + i, SIMDE_MM_HINT_T0);
		}

		isize row = start_row;
		while (row != last_row) {
			out_c[row] = in_c[perm_indices[row]];
			++row;
		}
		nrows += info.inc_n;
		start_row += info.inc_start;
	}
}

LDLT_MINSIZE
void apply_perm_rows<f32>::fn(
		f32* out,
		isize out_stride,
		f32 const* in,
		isize in_stride,
		isize n,
		i32 const* perm_indices,
		i32 sym) noexcept {

	auto info = detail::perm_helper(sym, n);

	isize n16 = n / 16 * 16;

	isize n_prefetch = n < 512 ? 0 : n16;

	isize start_row = 0;
	isize nrows = info.init;

	for (isize col = 0; col < n; ++col) {
		f32* out_c = out + out_stride * col;
		f32 const* in_c = in + in_stride * col;
		isize last_row = start_row + nrows;

		for (isize i = 0; i < n_prefetch; i += 16) {
			simde_mm_prefetch(in_c + i, SIMDE_MM_HINT_T0);
		}

		isize row = start_row;
		out_c += start_row;
		while (row != last_row) {
			i32 indices0 = perm_indices[row];
			*out_c = in_c[indices0];
			++out_c;
			++row;
		}
		nrows += info.inc_n;
		start_row += info.inc_start;
	}
}

} // namespace detail
} // namespace ldlt
