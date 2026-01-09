#pragma once

#if !defined(_MSC_VER)
#error "This file is only compatible with the MSVC compiler"
#endif

#include <cstdint>
#include <immintrin.h>

class uint128_t
{
public:
  uint64_t low;
  uint64_t high;

  // --- Constructors ---
  constexpr uint128_t()
    : low(0)
    , high(0)
  {
  }
  constexpr uint128_t(uint64_t l)
    : low(l)
    , high(0)
  {
  }
  constexpr uint128_t(uint64_t l, uint64_t h)
    : low(l)
    , high(h)
  {
  }

  // --- Type Conversions ---
  explicit operator bool() const { return low || high; }
  explicit operator uint64_t() const { return low; }
  explicit operator int64_t() const { return static_cast<int64_t>(low); }

  // --- Arithmetic Operators ---

  // Addition
  uint128_t operator+(const uint128_t& rhs) const
  {
    uint128_t result;
    unsigned char carry = _addcarry_u64(0, low, rhs.low, &result.low);
    _addcarry_u64(carry, high, rhs.high, &result.high);
    return result;
  }

  uint128_t& operator+=(const uint128_t& rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  // Subtraction
  uint128_t operator-(const uint128_t& rhs) const
  {
    uint128_t result;
    unsigned char borrow = _subborrow_u64(0, low, rhs.low, &result.low);
    _subborrow_u64(borrow, high, rhs.high, &result.high);
    return result;
  }

  uint128_t& operator-=(const uint128_t& rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  // Multiplication
  uint128_t operator*(const uint128_t& rhs) const
  {
    uint64_t product_high;
    uint64_t product_low = _umul128(low, rhs.low, &product_high);

    // The total high part is the high part of (low * rhs.low)
    // plus the cross terms (low * rhs.high) and (high * rhs.low)
    product_high += (low * rhs.high) + (high * rhs.low);

    return uint128_t(product_low, product_high);
  }

  uint128_t& operator*=(const uint128_t& rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  // Division (Note: Full 128-bit division is complex to implement purely with
  // intrinsics if the divisor is > 64 bits. This is a simplified version
  // handling common cases). For production-grade full 128/128 division, usage
  // of a library like Boost is strongly advised. However, if divisor fits in 64
  // bits, we can use _udiv128.
  uint128_t operator/(const uint128_t& rhs) const
  {
    if (rhs.high == 0) {
      // Optimization for 64-bit divisor
      uint64_t remainder;
      uint64_t quotient_high = 0; // High part of result
      uint64_t quotient_low;

      // If our high part is distinct, we divide the high part first
      if (high > 0) {
        // This is slightly tricky with _udiv128 directly as it does 128/64
        // -> 64. Standard long division algorithm is safer here for the general
        // implementation. For simplicity in this snippet, we will fallback to a
        // naive loop or simple approximation OR promote strictly the 64-bit
        // divisor case which is most common:

        quotient_high = high / rhs.low;
        uint64_t r_high = high % rhs.low;

        quotient_low = _udiv128(r_high, low, rhs.low, &remainder);
        return uint128_t(quotient_low, quotient_high);
      } else {
        return uint128_t(low / rhs.low, 0);
      }
    }
    // Fallback for full 128-bit divisor: Very slow basic binary long division
    if (rhs > *this)
      return uint128_t(0);
    if (rhs == *this)
      return uint128_t(1);

    uint128_t temp = *this;
    uint128_t quot = 0;
    uint128_t one = 1;

    // This is slow O(N) division, acceptable for simple utility, bad for heavy
    // math
    while (temp >= rhs) {
      // Find shift
      uint128_t shift_rhs = rhs;
      uint128_t shift_count = 1;
      while ((shift_rhs.high & 0x8000000000000000) == 0 &&
             (shift_rhs << 1) <= temp) {
        shift_rhs <<= 1;
        shift_count <<= 1;
      }
      temp -= shift_rhs;
      quot += shift_count;
    }
    return quot;
  }

  // Modulus
  uint128_t operator%(const uint128_t& rhs) const
  {
    return *this - (*this / rhs) * rhs;
  }

  uint128_t& operator%=(const uint128_t& rhs)
  {
    *this = *this % rhs;
    return *this;
  }

  // --- Bitwise Operators ---
  uint128_t operator<<(int shift) const
  {
    shift &= 127; // Mask the shift amount to imitate native hardware behavior
                  // (modulo 128)
    if (shift == 0)
      return *this;
    if (shift >= 64) {
      return uint128_t(0, low << (shift - 64));
    }
    return uint128_t((low << shift), (high << shift) | (low >> (64 - shift)));
  }

  uint128_t operator>>(int shift) const
  {
    shift &= 127; // Mask the shift amount to imitate native hardware behavior
                  // (modulo 128)
    if (shift == 0)
      return *this;
    if (shift >= 64) {
      return uint128_t(high >> (shift - 64), 0);
    }
    return uint128_t((low >> shift) | (high << (64 - shift)), (high >> shift));
  }

  // --- Shift by uint128_t Overloads ---
  uint128_t operator>>(const uint128_t& shift) const
  {
    // If shift amount is >= 128, the result behavior mimics hardware (modulo
    // 128)
    return *this >> static_cast<int>(shift.low);
  }

  uint128_t operator<<(const uint128_t& shift) const
  {
    // If shift amount is >= 128, the result behavior mimics hardware (modulo
    // 128)
    return *this << static_cast<int>(shift.low);
  }

  uint128_t& operator<<=(int shift)
  {
    *this = *this << shift;
    return *this;
  }
  uint128_t& operator>>=(int shift)
  {
    *this = *this >> shift;
    return *this;
  }

  uint128_t operator|(const uint128_t& rhs) const
  {
    return uint128_t(low | rhs.low, high | rhs.high);
  }
  uint128_t operator&(const uint128_t& rhs) const
  {
    return uint128_t(low & rhs.low, high & rhs.high);
  }
  uint128_t operator^(const uint128_t& rhs) const
  {
    return uint128_t(low ^ rhs.low, high ^ rhs.high);
  }
  uint128_t operator~() const { return uint128_t(~low, ~high); }

  // --- Comparison Operators ---
  bool operator==(const uint128_t& rhs) const
  {
    return low == rhs.low && high == rhs.high;
  }
  bool operator!=(const uint128_t& rhs) const { return !(*this == rhs); }
  bool operator<(const uint128_t& rhs) const
  {
    return high < rhs.high || (high == rhs.high && low < rhs.low);
  }
  bool operator>(const uint128_t& rhs) const { return rhs < *this; }
  bool operator<=(const uint128_t& rhs) const { return !(*this > rhs); }
  bool operator>=(const uint128_t& rhs) const { return !(*this < rhs); }
};
