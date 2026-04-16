//
// Copyright (c) 2023 INRIA
//
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <string>

#if defined(_MSC_VER)
#include <proxsuite/proxqp/utils/uint128_msvc.hpp>
using u128 = uint128_t;
#define MAKE_U128(low, high) u128(low, high)
#define CHECK_HIGH(val, expected) CHECK((val).high == (expected))
#define CHECK_LOW(val, expected) CHECK((val).low == (expected))

// Compile-time checks: verify constexpr operators are truly constexpr
static_assert(uint128_t(0) == uint128_t(0), "== must be constexpr");
static_assert(uint128_t(1) != uint128_t(2), "!= must be constexpr");
static_assert(uint128_t(1) < uint128_t(2), "<  must be constexpr");
static_assert(uint128_t(2) > uint128_t(1), ">  must be constexpr");
static_assert(uint128_t(1) <= uint128_t(1), "<= must be constexpr");
static_assert(uint128_t(1) >= uint128_t(1), ">= must be constexpr");
static_assert(static_cast<bool>(uint128_t(1)),
              "operator bool must be constexpr");
static_assert(!static_cast<bool>(uint128_t(0)),
              "operator bool(0) must be constexpr");
static_assert(static_cast<uint64_t>(uint128_t(42)) == 42,
              "operator uint64_t must be constexpr");
static_assert(static_cast<int64_t>(uint128_t(7)) == 7,
              "operator int64_t must be constexpr");
static_assert((uint128_t(0xFF) | uint128_t(0x100)) == uint128_t(0x1FF),
              "| must be constexpr");
static_assert((uint128_t(0xFF) & uint128_t(0x0F)) == uint128_t(0x0F),
              "& must be constexpr");
static_assert((uint128_t(0xFF) ^ uint128_t(0x0F)) == uint128_t(0xF0),
              "^ must be constexpr");
static_assert((~uint128_t(0)) ==
                uint128_t(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
              "~ must be constexpr");
static_assert((uint128_t(1) << 4) == uint128_t(16), "<< int must be constexpr");
static_assert((uint128_t(16) >> 4) == uint128_t(1), ">> int must be constexpr");
static_assert((uint128_t(1) << uint128_t(4)) == uint128_t(16),
              "<< u128 must be constexpr");
static_assert((uint128_t(16) >> uint128_t(4)) == uint128_t(1),
              ">> u128 must be constexpr");

#else
using u128 = __uint128_t;
#define MAKE_U128(low, high) ((u128(high) << 64) | low)
#define CHECK_HIGH(val, expected)                                              \
  CHECK(static_cast<uint64_t>(val >> 64) == (expected))
#define CHECK_LOW(val, expected) CHECK(static_cast<uint64_t>(val) == (expected))
#endif

std::ostream&
operator<<(std::ostream& os, u128 n)
{
  if (n == u128(0)) {
    return os << "0";
  }
  std::string s;
  while (n > u128(0)) {
    // Cast remainder to uint64_t for char addition.
    // Works for both custom struct (via explicit operator) and built-in type.
    u128 rem = n % u128(10);
    s += (char)('0' + static_cast<uint64_t>(rem));
    n = n / u128(10);
  }
  std::reverse(s.begin(), s.end());
  return os << s;
}

TEST_CASE("Constructors and Equality", "[uint128]")
{
  u128 zero(0);
  CHECK_LOW(zero, 0);
  CHECK_HIGH(zero, 0);

  u128 val(12345);
  CHECK_LOW(val, 12345);
  CHECK_HIGH(val, 0);

  u128 big = MAKE_U128(0xFFFFFFFFFFFFFFFF, 0x1);
  CHECK_LOW(big, 0xFFFFFFFFFFFFFFFF);
  CHECK_HIGH(big, 1);

  REQUIRE(u128(10) == u128(10));
  REQUIRE(u128(10) != u128(11));
}

TEST_CASE("Bitwise Shifts with u128 (The Fix)", "[uint128][shift]")
{
  // This is the specific case you asked for: u128 >> u128
  u128 val = u128(0);
  u128 shift_amt = u128(64);

  REQUIRE((val >> shift_amt) == u128(0));

  // Test shifting a real value by a u128
  u128 one(1);
  u128 two(2);
  REQUIRE((one << one) == u128(2)); // 1 << 1 = 2

  // Test large shift via u128
  u128 large_shift(100);
  u128 shifted = one << large_shift;

  // 1 << 100 results in high bit (1 << (100-64)) = 1 << 36
  CHECK_HIGH(shifted, (1ULL << 36));
  CHECK_LOW(shifted, 0);

#if defined(_MSC_VER)
  // Test over-shift (>= 128) via u128
  // Now MSVC behaves like __uint128_t (modulo 128 shift)
  u128 huge_shift(128);
  u128 pattern = MAKE_U128(0xFF, 0xFF);
  // pattern >> 128 is effectively pattern >> 0, which is pattern
  REQUIRE((pattern >> huge_shift) == pattern);
#endif
}

TEST_CASE("Standard Bitwise Shifts (int)", "[uint128][shift]")
{
  u128 val = MAKE_U128(1, 0); // low=1, high=0

  // Shift left crossing boundary
  u128 res = val << 64;
  CHECK_HIGH(res, 1);
  CHECK_LOW(res, 0);

  // Shift right crossing boundary
  u128 high_val = MAKE_U128(0, 1); // low=0, high=1
  res = high_val >> 64;
  CHECK_HIGH(res, 0);
  CHECK_LOW(res, 1);

  // Shift within high part
  u128 mix = MAKE_U128(0, 2);
  REQUIRE((mix << 1) == MAKE_U128(0, 4));
}

TEST_CASE("Arithmetic Operations", "[uint128][math]")
{
  SECTION("Addition")
  {
    u128 max_low = MAKE_U128(0xFFFFFFFFFFFFFFFF, 0);
    u128 one(1);
    u128 result = max_low + one;

    // Should carry over to high
    CHECK_HIGH(result, 1);
    CHECK_LOW(result, 0);
  }

  SECTION("Subtraction")
  {
    u128 zero = MAKE_U128(0, 0);
    u128 one(1);
    u128 result = zero - one;

    // Underflow checks
    CHECK_HIGH(result, 0xFFFFFFFFFFFFFFFF);
    CHECK_LOW(result, 0xFFFFFFFFFFFFFFFF);
  }

  SECTION("Multiplication")
  {
    u128 a(2);
    u128 b(3);
    REQUIRE((a * b) == u128(6));

    // Test overflow into high
    // 2^64 * 2 = 2^65
    // Construct 2^64 using the struct (low=0, high=1)
    u128 two_64 = MAKE_U128(0, 1);
    u128 two(2);
    u128 res = two_64 * two;
    CHECK_HIGH(res, 2);
    CHECK_LOW(res, 0);
  }

  SECTION("Division (Decimal Printing Helper)")
  {
    u128 a(100);
    u128 b(10);
    REQUIRE((a / b) == u128(10));

    // Test the optimize 64-bit divisor path
    u128 large = MAKE_U128(0, 1); // 2^64
    u128 two(2);
    u128 half_large = large / two; // 2^63
    CHECK_HIGH(half_large, 0);
    CHECK_LOW(half_large, (1ULL << 63));
  }
}

TEST_CASE("Comparison Operators", "[uint128][compare]")
{
  u128 small(10);
  u128 big(20);
  u128 huge = MAKE_U128(0, 1); // 2^64

  REQUIRE(small < big);
  REQUIRE(big > small);
  REQUIRE(huge > big);
  REQUIRE(small <= u128(10));
  REQUIRE(big >= u128(20));
}

TEST_CASE("Lehmer Constant Generation", "[uint128][lehmer]")
{
  constexpr u128 lehmer64_constant(0xda942042e4dd58b5);
  CHECK_LOW(lehmer64_constant, 0xda942042e4dd58b5);
  CHECK_HIGH(lehmer64_constant, 0);
}

TEST_CASE("String Output (Decimal)", "[uint128][print]")
{
  std::stringstream ss;

  u128 val(12345);
  ss << val;
  REQUIRE(ss.str() == "12345");

  ss.str(""); // Clear
  u128 zero(0);
  ss << zero;
  REQUIRE(ss.str() == "0");

  // Test value larger than uint64_t max (18446744073709551615)
  // 18446744073709551616 is 2^64 (high=1, low=0)
  ss.str("");
  u128 big = MAKE_U128(0, 1);
  ss << big;
  REQUIRE(ss.str() == "18446744073709551616");
}

TEST_CASE("Division by Zero Guard", "[uint128][division][error]")
{
#if defined(_MSC_VER)
  u128 numerator(100);
  u128 zero(0);

  // Division by zero should throw std::domain_error
  REQUIRE_THROWS_AS(numerator / zero, std::domain_error);

  // Test with zero constructed from MAKE_U128
  u128 zero_via_macro = MAKE_U128(0, 0);
  REQUIRE_THROWS_AS(numerator / zero_via_macro, std::domain_error);

  // Valid division should not throw
  u128 ten(10);
  REQUIRE_NOTHROW(numerator / ten);
#endif
}

TEST_CASE("Shift by uint128_t with shift.low > INT_MAX",
          "[uint128][shift][large_shift_amount]")
{
#if defined(_MSC_VER)
  u128 one(1);
  u128 shift_large(0x80000003ULL);
  REQUIRE((one << shift_large) == u128(8));
  REQUIRE((u128(8) >> shift_large) == u128(1));

  u128 shift_zero_bits(0xFFFFFFFF80000000ULL);
  u128 val = MAKE_U128(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);
  REQUIRE((val << shift_zero_bits) == val);
  REQUIRE((val >> shift_zero_bits) == val);
#endif
}

TEST_CASE("Compound Shift Operators with uint128_t",
          "[uint128][shift][compound]")
{
  SECTION("Left Shift Compound Operator (uint128_t)")
  {
    u128 val = u128(1);
    u128 shift_amt = u128(1);

    // val <<= shift_amt
    val <<= shift_amt;
    REQUIRE(val == u128(2)); // 1 << 1 = 2

    // Test crossing boundary
    u128 val2 = u128(1);
    u128 shift_64 = u128(64);
    val2 <<= shift_64;
    CHECK_HIGH(val2, 1);
    CHECK_LOW(val2, 0);
  }

  SECTION("Right Shift Compound Operator (uint128_t)")
  {
    u128 val = MAKE_U128(0, 1); // high=1, low=0 (represents 2^64)
    u128 shift_amt = u128(1);

    // val >>= shift_amt
    val >>= shift_amt;
    CHECK_HIGH(val, 0);
    CHECK_LOW(val, (1ULL << 63)); // 2^63

    // Test crossing boundary with larger shift
    u128 val2 = MAKE_U128(0, 1);
    u128 shift_64 = u128(64);
    val2 >>= shift_64;
    CHECK_HIGH(val2, 0);
    CHECK_LOW(val2, 1);
  }

  SECTION("Chained Shift Operations (uint128_t)")
  {
    u128 val = u128(1);
    u128 shift1 = u128(3);
    u128 shift2 = u128(2);

    // (1 << 3) << 2 = 1 << 5 = 32
    val <<= shift1;
    val <<= shift2;
    REQUIRE(val == u128(32));
  }

  SECTION("Large Shift via uint128_t")
  {
    u128 one(1);
    u128 shift_100(100);

    one <<= shift_100;
    // 1 << 100 results in high bit (1 << (100-64)) = 1 << 36
    CHECK_HIGH(one, (1ULL << 36));
    CHECK_LOW(one, 0);
  }

  SECTION("Over-shift Behavior (uint128_t)")
  {
#if defined(_MSC_VER)
    u128 pattern = MAKE_U128(0xFF, 0xFF);
    u128 huge_shift(128);

    // pattern <<= 128 should wrap (modulo 128)
    pattern <<= huge_shift;
    REQUIRE(pattern == MAKE_U128(0xFF, 0xFF)); // No effective change

    // Test right shift over-shift
    u128 pattern2 = MAKE_U128(0xFF, 0xFF);
    pattern2 >>= huge_shift;
    REQUIRE(pattern2 == MAKE_U128(0xFF, 0xFF)); // No effective change
#endif
  }
}
