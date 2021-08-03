#ifndef LDLT_SIMD_HPP_RXCUB3KZS
#define LDLT_SIMD_HPP_RXCUB3KZS

#include "ldlt/detail/macros.hpp"
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>

namespace ldlt {

using usize = decltype(sizeof(0));
using f32 = float;
using f64 = double;

static_assert(static_cast<unsigned char>(-1) == 255, "char should have 8 bits");
static_assert(sizeof(f32) == 4, "f32 should be 32 bits");
static_assert(sizeof(f64) == 8, "f64 should be 64 bits");

namespace detail {

#define LDLT_REMOVE_PAREN(...) __VA_ARGS__
#define LDLT_NOM_SEMICOLON static_assert(true, ".");

#define LDLT_FN_IMPL(Fn, Prefix, Suffix)                                       \
	LDLT_INLINE auto Fn(Pack rhs) const noexcept->Pack {                         \
		return Pack{simde_mm##Prefix##_##Fn##_##Suffix(inner, rhs.inner)};         \
	}                                                                            \
	LDLT_NOM_SEMICOLON

#define LDLT_FN_IMPL3(Fn, Prefix, Suffix)                                      \
	LDLT_INLINE static auto Fn(Pack a, Pack b, Pack c) noexcept->Pack {          \
		return Pack{                                                               \
				simde_mm##Prefix##_##Fn##_##Suffix(a.inner, b.inner, c.inner)};        \
	}                                                                            \
	LDLT_NOM_SEMICOLON

#define LDLT_ARITHMETIC_IMPL(Prefix, Suffix)                                   \
	LDLT_FN_IMPL(add, Prefix, Suffix);                                           \
	LDLT_FN_IMPL(sub, Prefix, Suffix);                                           \
	LDLT_FN_IMPL(mul, Prefix, Suffix);                                           \
	LDLT_FN_IMPL(div, Prefix, Suffix);                                           \
	LDLT_FN_IMPL3(fmadd, Prefix, Suffix);  /* (a * b + c) */                     \
	LDLT_FN_IMPL3(fmsub, Prefix, Suffix)   /* (a * b - c) */                     \
	LDLT_FN_IMPL3(fnmadd, Prefix, Suffix); /* (-a * b + c) */                    \
	LDLT_FN_IMPL3(fnmsub, Prefix, Suffix)  /* (-a * b - c) */

#define LDLT_LOAD_STORE(Prefix, Suffix)                                        \
	LDLT_INLINE static auto load_unaligned(                                      \
			ScalarType const* ptr) noexcept->Pack {                                  \
		return Pack{simde_mm##Prefix##_loadu_##Suffix(ptr)};                       \
	}                                                                            \
	LDLT_INLINE static auto zero() noexcept->Pack {                              \
		return Pack{simde_mm##Prefix##_setzero_##Suffix()};                        \
	}                                                                            \
	LDLT_INLINE static auto broadcast(ScalarType value) noexcept->Pack {         \
		return Pack{simde_mm##Prefix##_set1_##Suffix(value)};                      \
	}                                                                            \
	LDLT_INLINE void store_unaligned(ScalarType* ptr) const noexcept {           \
		simde_mm##Prefix##_storeu_##Suffix(ptr, inner);                            \
	}                                                                            \
	LDLT_NOM_SEMICOLON

template <typename T, usize N>
struct Pack;

template <>
struct Pack<f32, 4> {
	using ScalarType = f32;

	simde__m128 inner;
	LDLT_ARITHMETIC_IMPL(, ps);
	LDLT_LOAD_STORE(, ps);

	LDLT_INLINE auto sum() const noexcept -> f32 {
		// inner = {0 1 2 3}
		simde__m128 dup = simde_mm_movehdup_ps(inner);
		// r1 = {1 1 3 3}
		simde__m128 sum2 = simde_mm_add_ps(inner, dup);
		// r2 = {01 11 23 33}
		simde__m128 shuf = simde_mm_movehl_ps(sum2, sum2);
		// r3 = {23 33 23 33}
		simde__m128 sum4 = simde_mm_add_ss(sum2, shuf);
		// r3 = {0123 X X X}

		return simde_mm_cvtss_f32(sum4);
	}
};

template <>
struct Pack<f32, 8> {
	using ScalarType = f32;

	simde__m256 inner;
	LDLT_ARITHMETIC_IMPL(256, ps);
	LDLT_LOAD_STORE(256, ps);

	LDLT_INLINE auto lo() const noexcept -> Pack<f32, 4> {
		return {simde_mm256_castps256_ps128(inner)};
	}
	LDLT_INLINE auto hi() const noexcept -> Pack<f32, 4> {
		return {simde_mm256_extractf128_ps(inner, 1)};
	}

	LDLT_INLINE auto sum() const noexcept -> f32 { return lo().add(hi()).sum(); }
};
template <>
struct Pack<f32, 16> {
	using ScalarType = f32;

	simde__m512 inner;
	LDLT_ARITHMETIC_IMPL(512, ps);
	LDLT_LOAD_STORE(512, ps);

	LDLT_INLINE auto lo() const noexcept -> Pack<f32, 8> {
		return {simde_mm512_castps512_ps256(inner)};
	}
	LDLT_INLINE auto hi() const noexcept -> Pack<f32, 8> {
		return {simde_mm256_castpd_ps(
				simde_mm512_extractf64x4_pd(simde_mm512_castps_pd(inner), 1))};
	}
	LDLT_INLINE auto sum() const noexcept -> f32 { return lo().add(hi()).sum(); }
};

template <>
struct Pack<f64, 2> {
	using ScalarType = f64;

	simde__m128d inner;
	LDLT_ARITHMETIC_IMPL(, pd);
	LDLT_LOAD_STORE(, pd);

	LDLT_INLINE auto sum() const noexcept -> f64 {
		simde__m128 inner_ps = simde_mm_castpd_ps(inner);
		// inner = {0 1}
		simde__m128d shuf =
				simde_mm_castps_pd(simde_mm_movehl_ps(inner_ps, inner_ps));
		// shuf = {1 1}
		simde__m128d sum2 = simde_mm_add_sd(inner, shuf);
		return simde_mm_cvtsd_f64(sum2);
	}
};
template <>
struct Pack<f64, 4> {
	using ScalarType = f64;

	simde__m256d inner;
	LDLT_ARITHMETIC_IMPL(256, pd);
	LDLT_LOAD_STORE(256, pd);

	LDLT_INLINE auto lo() const noexcept -> Pack<f64, 2> {
		return {simde_mm256_castpd256_pd128(inner)};
	}
	LDLT_INLINE auto hi() const noexcept -> Pack<f64, 2> {
		return {simde_mm256_extractf128_pd(inner, 1)};
	}
	LDLT_INLINE auto sum() const noexcept -> f64 { return lo().add(hi()).sum(); }
};
template <>
struct Pack<f64, 8> {
	using ScalarType = f64;

	simde__m512d inner;
	LDLT_ARITHMETIC_IMPL(512, pd);
	LDLT_LOAD_STORE(512, pd);

	LDLT_INLINE auto lo() const noexcept -> Pack<f64, 4> {
		return {simde_mm512_castpd512_pd256(inner)};
	}
	LDLT_INLINE auto hi() const noexcept -> Pack<f64, 4> {
		return {simde_mm512_extractf64x4_pd(inner, 1)};
	}
	LDLT_INLINE auto sum() const noexcept -> f64 { return lo().add(hi()).sum(); }
};

template <typename T>
struct NativePackInfo {
	using Type = void;
};

template <>
struct NativePackInfo<f32> {
	static constexpr usize N = SIMDE_NATURAL_VECTOR_SIZE / 32;
	using Type = Pack<f32, N>;
};
template <>
struct NativePackInfo<f64> {
	static constexpr usize N = SIMDE_NATURAL_VECTOR_SIZE / 64;
	using Type = Pack<f64, N>;
};

template <typename T>
using NativePack = typename NativePackInfo<T>::Type;

} // namespace detail
} // namespace ldlt

#endif /* end of include guard LDLT_SIMD_HPP_RXCUB3KZS */
