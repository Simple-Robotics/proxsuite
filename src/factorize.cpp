#include <ldlt/factorize.hpp>

namespace ldlt {
namespace detail {
LDLT_EXPLICIT_TPL_DEF(1, factorize_unblocked<f32>);
LDLT_EXPLICIT_TPL_DEF(2, factorize_blocked<f32>);
LDLT_EXPLICIT_TPL_DEF(1, factorize_unblocked<f64>);
LDLT_EXPLICIT_TPL_DEF(2, factorize_blocked<f64>);
LDLT_EXPLICIT_TPL_DEF(3, compute_permutation<f32>);
LDLT_EXPLICIT_TPL_DEF(3, compute_permutation<f64>);

#if defined(SIMDE_X86_AVX512F_NATIVE)
// TODO: avx512 implementation
#elif defined(SIMDE_X86_AVX2_NATIVE)

// avx256 implementation
void apply_perm_rows<f64>::fn(
		f64* HEDLEY_RESTRICT out,
		isize out_stride,
		f64 const* HEDLEY_RESTRICT in,
		isize in_stride,
		isize n,
		i32 const* HEDLEY_RESTRICT perm_indices) noexcept {

	isize n32 = n / 32 * 32;
	isize n16 = n / 16 * 16;
	isize n8 = n / 8 * 8;
	isize n4 = n / 4 * 4;
	isize n2 = n / 2 * 2;

	isize row = 0;
	for (; row < n32; row += 32) {
		simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row + 0);
		simde__m128i indices1 = simde_mm_loadu_epi32(perm_indices + row + 4);
		simde__m128i indices2 = simde_mm_loadu_epi32(perm_indices + row + 8);
		simde__m128i indices3 = simde_mm_loadu_epi32(perm_indices + row + 12);
		simde__m128i indices4 = simde_mm_loadu_epi32(perm_indices + row + 16);
		simde__m128i indices5 = simde_mm_loadu_epi32(perm_indices + row + 20);
		simde__m128i indices6 = simde_mm_loadu_epi32(perm_indices + row + 24);
		simde__m128i indices7 = simde_mm_loadu_epi32(perm_indices + row + 28);

		i32 _prefetch_indices[32];
		i32* prefetch_indices = &_prefetch_indices[0];
		std::memcpy(prefetch_indices, perm_indices + row, sizeof(i32) * 32);

		for (isize col = 0; col < n; ++col) {
			f64 const* base = in + in_stride * col;

			isize offset = 0;
			if ((col + offset) < n) {
				for (isize i = 0; i < 32; ++i) {
					f64 const* offset_base = in + in_stride * (col + offset);
					simde_mm_prefetch(
							offset_base + prefetch_indices[i], SIMDE_MM_HINT_T0);
				}
			}

			simde__m256d p0 = simde_mm256_i32gather_pd(base, indices0, 8);
			simde__m256d p1 = simde_mm256_i32gather_pd(base, indices1, 8);
			simde__m256d p2 = simde_mm256_i32gather_pd(base, indices2, 8);
			simde__m256d p3 = simde_mm256_i32gather_pd(base, indices3, 8);
			simde__m256d p4 = simde_mm256_i32gather_pd(base, indices4, 8);
			simde__m256d p5 = simde_mm256_i32gather_pd(base, indices5, 8);
			simde__m256d p6 = simde_mm256_i32gather_pd(base, indices6, 8);
			simde__m256d p7 = simde_mm256_i32gather_pd(base, indices7, 8);

			simde_mm256_storeu_pd(out + out_stride * col + 0, p0);
			simde_mm256_storeu_pd(out + out_stride * col + 4, p1);
			simde_mm256_storeu_pd(out + out_stride * col + 8, p2);
			simde_mm256_storeu_pd(out + out_stride * col + 12, p3);
			simde_mm256_storeu_pd(out + out_stride * col + 16, p4);
			simde_mm256_storeu_pd(out + out_stride * col + 20, p5);
			simde_mm256_storeu_pd(out + out_stride * col + 24, p6);
			simde_mm256_storeu_pd(out + out_stride * col + 28, p7);
		}
		out += 32;
	}
	if (row != n16) {
		simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row + 0);
		simde__m128i indices1 = simde_mm_loadu_epi32(perm_indices + row + 4);
		simde__m128i indices2 = simde_mm_loadu_epi32(perm_indices + row + 8);
		simde__m128i indices3 = simde_mm_loadu_epi32(perm_indices + row + 12);

		for (isize col = 0; col < n; ++col) {
			f64 const* base = in + in_stride * col;

			simde__m256d p0 = simde_mm256_i32gather_pd(base, indices0, 8);
			simde__m256d p1 = simde_mm256_i32gather_pd(base, indices1, 8);
			simde__m256d p2 = simde_mm256_i32gather_pd(base, indices2, 8);
			simde__m256d p3 = simde_mm256_i32gather_pd(base, indices3, 8);

			simde_mm256_storeu_pd(out + out_stride * col + 0, p0);
			simde_mm256_storeu_pd(out + out_stride * col + 4, p1);
			simde_mm256_storeu_pd(out + out_stride * col + 8, p2);
			simde_mm256_storeu_pd(out + out_stride * col + 12, p3);
		}
		out += 16;
		row += 16;
	}
	if (row != n8) {
		simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row + 0);
		simde__m128i indices1 = simde_mm_loadu_epi32(perm_indices + row + 4);

		for (isize col = 0; col < n; ++col) {
			f64 const* base = in + in_stride * col;

			simde__m256d p0 = simde_mm256_i32gather_pd(base, indices0, 8);
			simde__m256d p1 = simde_mm256_i32gather_pd(base, indices1, 8);

			simde_mm256_storeu_pd(out + out_stride * col + 0, p0);
			simde_mm256_storeu_pd(out + out_stride * col + 4, p1);
		}
		out += 8;
		row += 8;
	}
	if (row != n4) {
		simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row + 0);
		for (isize col = 0; col < n; ++col) {
			f64 const* base = in + in_stride * col;
			simde__m256d p0 = simde_mm256_i32gather_pd(base, indices0, 8);
			simde_mm256_storeu_pd(out + out_stride * col + 0, p0);
		}
		out += 4;
		row += 4;
	}
	if (row != n2) {
		simde__m128i indices0{};
		std::memcpy(&indices0, perm_indices + row, sizeof(*perm_indices) * 2);

		for (isize col = 0; col < n; ++col) {
			f64 const* base = in + in_stride * col;
			simde__m128d p0 = simde_mm_i32gather_pd(base, indices0, 8);
			simde_mm_storeu_pd(out + out_stride * col + 0, p0);
		}
		out += 2;
		row += 2;
	}
	if (row != n) {
		i32 indices0 = perm_indices[row];
		for (isize col = 0; col < n; ++col) {
			out[out_stride * col] = in[in_stride * col + indices0];
		}
	}
}

void apply_perm_rows<f32>::fn(
		f32* out,
		isize out_stride,
		f32 const* in,
		isize in_stride,
		isize n,
		i32 const* perm_indices) noexcept {

	isize n64 = n / 64 * 64;
	isize n32 = n / 32 * 32;
	isize n16 = n / 16 * 16;
	isize n8 = n / 8 * 8;
	isize n4 = n / 4 * 4;
	isize n2 = n / 2 * 2;

	isize row = 0;
	for (; row < n64; row += 64) {
		simde__m256i indices0 = simde_mm256_loadu_epi32(perm_indices + row + 0);
		simde__m256i indices1 = simde_mm256_loadu_epi32(perm_indices + row + 8);
		simde__m256i indices2 = simde_mm256_loadu_epi32(perm_indices + row + 16);
		simde__m256i indices3 = simde_mm256_loadu_epi32(perm_indices + row + 24);
		simde__m256i indices4 = simde_mm256_loadu_epi32(perm_indices + row + 32);
		simde__m256i indices5 = simde_mm256_loadu_epi32(perm_indices + row + 40);
		simde__m256i indices6 = simde_mm256_loadu_epi32(perm_indices + row + 48);
		simde__m256i indices7 = simde_mm256_loadu_epi32(perm_indices + row + 56);

		i32 _prefetch_indices[64];
		i32* prefetch_indices = &_prefetch_indices[0];
		std::memcpy(prefetch_indices, perm_indices + row, sizeof(i32) * 64);

		for (isize col = 0; col < n; ++col) {
			f32 const* base = in + in_stride * col;

			isize offset = 0;
			if ((col + offset) < n) {
				for (isize i = 0; i < 64; ++i) {
					f32 const* offset_base = in + in_stride * (col + offset);
					simde_mm_prefetch(
							offset_base + prefetch_indices[i], SIMDE_MM_HINT_T0);
				}
			}

			simde__m256 p0 = simde_mm256_i32gather_ps(base, indices0, 4);
			simde__m256 p1 = simde_mm256_i32gather_ps(base, indices1, 4);
			simde__m256 p2 = simde_mm256_i32gather_ps(base, indices2, 4);
			simde__m256 p3 = simde_mm256_i32gather_ps(base, indices3, 4);
			simde__m256 p4 = simde_mm256_i32gather_ps(base, indices4, 4);
			simde__m256 p5 = simde_mm256_i32gather_ps(base, indices5, 4);
			simde__m256 p6 = simde_mm256_i32gather_ps(base, indices6, 4);
			simde__m256 p7 = simde_mm256_i32gather_ps(base, indices7, 4);

			simde_mm256_storeu_ps(out + out_stride * col + 0, p0);
			simde_mm256_storeu_ps(out + out_stride * col + 8, p1);
			simde_mm256_storeu_ps(out + out_stride * col + 16, p2);
			simde_mm256_storeu_ps(out + out_stride * col + 24, p3);
			simde_mm256_storeu_ps(out + out_stride * col + 32, p4);
			simde_mm256_storeu_ps(out + out_stride * col + 40, p5);
			simde_mm256_storeu_ps(out + out_stride * col + 48, p6);
			simde_mm256_storeu_ps(out + out_stride * col + 56, p7);
		}
		out += 64;
	}
	if (row != n32) {
		simde__m256i indices0 = simde_mm256_loadu_epi32(perm_indices + row + 0);
		simde__m256i indices1 = simde_mm256_loadu_epi32(perm_indices + row + 8);
		simde__m256i indices2 = simde_mm256_loadu_epi32(perm_indices + row + 16);
		simde__m256i indices3 = simde_mm256_loadu_epi32(perm_indices + row + 24);

		for (isize col = 0; col < n; ++col) {
			f32 const* base = in + in_stride * col;

			simde__m256 p0 = simde_mm256_i32gather_ps(base, indices0, 4);
			simde__m256 p1 = simde_mm256_i32gather_ps(base, indices1, 4);
			simde__m256 p2 = simde_mm256_i32gather_ps(base, indices2, 4);
			simde__m256 p3 = simde_mm256_i32gather_ps(base, indices3, 4);

			simde_mm256_storeu_ps(out + out_stride * col + 0, p0);
			simde_mm256_storeu_ps(out + out_stride * col + 8, p1);
			simde_mm256_storeu_ps(out + out_stride * col + 16, p2);
			simde_mm256_storeu_ps(out + out_stride * col + 24, p3);
		}
		out += 32;
		row += 32;
	}
	if (row != n16) {
		simde__m256i indices0 = simde_mm256_loadu_epi32(perm_indices + row + 0);
		simde__m256i indices1 = simde_mm256_loadu_epi32(perm_indices + row + 8);

		for (isize col = 0; col < n; ++col) {
			f32 const* base = in + in_stride * col;

			simde__m256 p0 = simde_mm256_i32gather_ps(base, indices0, 4);
			simde__m256 p1 = simde_mm256_i32gather_ps(base, indices1, 4);

			simde_mm256_storeu_ps(out + out_stride * col + 0, p0);
			simde_mm256_storeu_ps(out + out_stride * col + 8, p1);
		}
		out += 16;
		row += 16;
	}
	if (row != n8) {
		simde__m256i indices0 = simde_mm256_loadu_epi32(perm_indices + row + 0);
		for (isize col = 0; col < n; ++col) {
			f32 const* base = in + in_stride * col;
			simde__m256 p0 = simde_mm256_i32gather_ps(base, indices0, 4);
			simde_mm256_storeu_ps(out + out_stride * col + 0, p0);
		}
		out += 8;
		row += 8;
	}
	if (row != n4) {
		simde__m128i indices0 = simde_mm_loadu_epi32(perm_indices + row + 0);
		for (isize col = 0; col < n; ++col) {
			f32 const* base = in + in_stride * col;
			simde__m128 p0 = simde_mm_i32gather_ps(base, indices0, 4);
			simde_mm_storeu_ps(out + out_stride * col + 0, p0);
		}
		out += 4;
		row += 4;
	}
	if (row != n2) {
		simde__m128i indices0;
		std::memcpy(&indices0, perm_indices + row + 0, sizeof(*perm_indices) * 2);
		std::memcpy(
				static_cast<unsigned char*>(static_cast<void*>(&indices0)) +
						2 * sizeof(i32),
				perm_indices + row + 0,
				sizeof(i32) * 2);

		for (isize col = 0; col < n; ++col) {
			f32 const* base = in + in_stride * col;
			simde__m128 p0 = simde_mm_i32gather_ps(base, indices0, 4);
			std::memcpy(out + out_stride * col, &p0, sizeof(f32) * 2);
		}
		out += 2;
		row += 2;
	}
	if (row != n) {
		i32 indices0 = perm_indices[row];
		for (isize col = 0; col < n; ++col) {
			out[out_stride * col] = in[in_stride * col + indices0];
		}
	}
}
#else
// unvectorized implementation
// TODO: benchmark and optimize?

void apply_perm_rows<f64>::fn(
		f64* out,
		isize out_stride,
		f64 const* in,
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
void apply_perm_rows<f32>::fn(
		f32* out,
		isize out_stride,
		f32 const* in,
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
#endif

} // namespace detail
} // namespace ldlt
