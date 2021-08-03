#ifndef LDLT_SIMD_HPP_RXCUB3KZS
#define LDLT_SIMD_HPP_RXCUB3KZS

#include "ldlt/detail/macros.hpp"
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>

namespace ldlt {
using usize = decltype(sizeof(0));
using f32 = float;
using f64 = double;

namespace detail {

#define LDLT_REMOVE_PAREN(...) __VA_ARGS__
#define LDLT_NOM_SEMICOLON static_assert(true, ".");

#define LDLT_FN_IMPL(Fn, Prefix, Suffix)                                       \
	LDLT_INLINE auto Fn(Pack rhs) const noexcept->Pack {                         \
		return Pack{simde_mm##Prefix##_##Fn##_##Suffix(inner, rhs.inner)};         \
	}                                                                            \
	LDLT_NOM_SEMICOLON

#define LDLT_ARITHMETIC_IMPL(Prefix, Suffix)                                   \
	LDLT_FN_IMPL(add, Prefix, Suffix);                                           \
	LDLT_FN_IMPL(sub, Prefix, Suffix);                                           \
	LDLT_FN_IMPL(mul, Prefix, Suffix);                                           \
	LDLT_FN_IMPL(div, Prefix, Suffix)

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
		simde__m128 lo = simde_mm_unpacklo_ps(inner, inner);
		simde__m128 hi = simde_mm_unpackhi_ps(inner, inner);
		simde__m128 lohi = simde_mm_add_ps(lo, hi);
		return simde_mm_extract_ps(lohi, 0) + simde_mm_extract_ps(lohi, 2);
	}
};
template <>
struct Pack<f32, 8> {
	using ScalarType = f32;

	simde__m256 inner;
	LDLT_ARITHMETIC_IMPL(256, ps);
	LDLT_LOAD_STORE(256, ps);
};
template <>
struct Pack<f32, 16> {
	using ScalarType = f32;

	simde__m512 inner;
	LDLT_ARITHMETIC_IMPL(512, ps);
	LDLT_LOAD_STORE(512, ps);
};

template <>
struct Pack<f64, 2> {
	using ScalarType = f64;

	simde__m128d inner;
	LDLT_ARITHMETIC_IMPL(, pd);
	LDLT_LOAD_STORE(, pd);
};
template <>
struct Pack<f64, 4> {
	using ScalarType = f64;

	simde__m256d inner;
	LDLT_ARITHMETIC_IMPL(256, pd);
	LDLT_LOAD_STORE(256, pd);
};
template <>
struct Pack<f64, 8> {
	using ScalarType = f64;

	simde__m512d inner;
	LDLT_ARITHMETIC_IMPL(512, pd);
	LDLT_LOAD_STORE(512, pd);
};

template <typename T>
struct NativePackInfo {
	using Type = void;
};

template <>
struct NativePackInfo<f32> {
	static constexpr usize N = SIMDE_NATURAL_VECTOR_SIZE / CHAR_BIT / sizeof(f32);
	using Type = Pack<f32, N>;
};
template <>
struct NativePackInfo<f64> {
	static constexpr usize N = SIMDE_NATURAL_VECTOR_SIZE / CHAR_BIT / sizeof(f64);
	using Type = Pack<f64, N>;
};

template <typename T>
using NativePack = typename NativePackInfo<T>::Type;

} // namespace detail
} // namespace ldlt

#endif /* end of include guard LDLT_SIMD_HPP_RXCUB3KZS */
