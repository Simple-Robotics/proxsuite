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

// avx256 implementation
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

	isize n32 = n / 32 * 32;
	isize n16 = n / 16 * 16;
	isize n8 = n / 8 * 8;
	isize n4 = n / 4 * 4;
	isize n2 = n / 2 * 2;

	isize n_prefetch = n < 512 ? 0 : n8;

	(void)n32, (void)n16, (void)n8, (void)n4, (void)n2;

	for (isize col = 0; col < n; ++col) {
		f64* out_c = out + out_stride * col;
		f64 const* in_c = in + in_stride * col;

		for (isize i = 0; i < n_prefetch; i += 8) {
			simde_mm_prefetch(in_c + i, SIMDE_MM_HINT_T0);
		}

		// TODO: benchmark on more modern cpu. gathers may have improved
#if 0 && defined(SIMDE_X86_AVX2_NATIVE)
		isize row = info.init / 4 * 4;

		while (row != n4) {
			simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row);
			simde__m256d p0 = simde_mm256_i32gather_pd(in_c, indices0, 8);
			simde_mm256_storeu_pd(out_c, p0);
			out_c += 4;
			row += 4;
		}
		while (row != n2) {
			simde__m128i indices0{};
			std::memcpy(&indices0, perm_indices + row, sizeof(i32) * 2);
			simde__m128d p0 = simde_mm_i32gather_pd(in_c, indices0, 8);
			simde_mm_storeu_pd(out_c, p0);
			out_c += 2;
			row += 2;
		}

#else
		isize row = info.init;
#endif

		while (row != n) {
			i32 indices0 = perm_indices[row];
			*out_c = in_c[indices0];

			++out_c;
			++row;
		}
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

	isize n64 = n / 64 * 64;
	isize n32 = n / 32 * 32;
	isize n16 = n / 16 * 16;
	isize n8 = n / 8 * 8;
	isize n4 = n / 4 * 4;
	isize n2 = n / 2 * 2;
	(void)n64, (void)n32, (void)n16, (void)n8, (void)n4, (void)n2;

	isize n_prefetch = n < 512 ? 0 : n16;

	for (isize col = 0; col < n; ++col) {
		f32* out_c = out + out_stride * col;
		f32 const* in_c = in + in_stride * col;

		for (isize i = 0; i < n_prefetch; i += 16) {
			simde_mm_prefetch(in_c + i, SIMDE_MM_HINT_T0);
		}

#if 0 && defined(SIMDE_X86_AVX2_NATIVE)
		isize row = info.init / 8 * 8;

		while (row != n8) {
			simde__m256i indices0 = simde_mm256_loadu_epi32(perm_indices + row);
			simde__m256 p0 = simde_mm256_i32gather_ps(in_c, indices0, 4);

			simde_mm256_storeu_ps(out_c, p0);
			out_c += 8;
			row += 8;
		}
		while (row != n4) {
			simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row);
			simde__m128 p0 = simde_mm_i32gather_ps(in_c, indices0, 8);
			simde_mm_storeu_ps(out_c, p0);
			out_c += 4;
			row += 4;
		}
#else
		isize row = info.init;
#endif
		while (row != n) {
			i32 indices0 = perm_indices[row];
			*out = in_c[indices0];
			++out_c;
			++row;
		}
    info.init += info.inc;
	}
}

} // namespace detail
} // namespace ldlt
