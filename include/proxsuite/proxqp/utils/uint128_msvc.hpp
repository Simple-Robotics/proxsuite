#pragma once

#if !defined(_MSC_VER)
#error "This file is only compatible with the MSVC compiler"
#endif

#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

class uint128_t
{
public:
  uint64_t low = 0;
  uint64_t high = 0;

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
  constexpr explicit operator bool() const { return low || high; }
  constexpr explicit operator uint64_t() const { return low; }
  constexpr explicit operator int64_t() const
  {
    return static_cast<int64_t>(low);
  }

  // --- Arithmetic Operators ---

  uint128_t operator+(const uint128_t& rhs) const
  {
    uint128_t result = 0;
    const unsigned char carry = _addcarry_u64(0, low, rhs.low, &result.low);
    std::ignore = _addcarry_u64(carry, high, rhs.high, &result.high);
    return result;
  }

  uint128_t& operator+=(const uint128_t& rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  uint128_t operator-(const uint128_t& rhs) const
  {
    uint128_t result = 0;
    const unsigned char borrow = _subborrow_u64(0, low, rhs.low, &result.low);
    _subborrow_u64(borrow, high, rhs.high, &result.high);
    return result;
  }

  uint128_t& operator-=(const uint128_t& rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  uint128_t operator*(const uint128_t& rhs) const
  {
    uint64_t product_high = 0;
    const uint64_t product_low = _umul128(low, rhs.low, &product_high);
    product_high += (low * rhs.high) + (high * rhs.low);
    return uint128_t(product_low, product_high);
  }

  uint128_t& operator*=(const uint128_t& rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  // Division by a 64-bit divisor uses _udiv128. For a full 128-bit divisor,
  // falls back to binary long division.
  uint128_t operator/(const uint128_t& rhs) const
  {
    if (!rhs) {
      throw std::domain_error("uint128 division by zero");
    }
    if (rhs.high == 0) {
      if (high > 0) {
        const uint64_t quotient_high = high / rhs.low;
        const uint64_t r_high = high % rhs.low;
        uint64_t remainder = 0;
        const uint64_t quotient_low =
          _udiv128(r_high, low, rhs.low, &remainder);
        return uint128_t(quotient_low, quotient_high);
      }
      return uint128_t(low / rhs.low, 0);
    }
    // Binary long division for 128-bit divisor
    if (rhs > *this)
      return uint128_t(0);
    if (rhs == *this)
      return uint128_t(1);

    uint128_t temp = *this;
    uint128_t quot = 0;
    while (temp >= rhs) {
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
  constexpr uint128_t operator<<(int shift) const
  {
    shift &= 127; // wrap modulo 128, matching hardware behavior
    if (shift == 0)
      return *this;
    if (shift >= 64) {
      return uint128_t(0, low << (shift - 64));
    }
    return uint128_t((low << shift), (high << shift) | (low >> (64 - shift)));
  }

  constexpr uint128_t operator>>(int shift) const
  {
    shift &= 127; // wrap modulo 128, matching hardware behavior
    if (shift == 0)
      return *this;
    if (shift >= 64) {
      return uint128_t(high >> (shift - 64), 0);
    }
    return uint128_t((low >> shift) | (high << (64 - shift)), (high >> shift));
  }

  // --- Shift by uint128_t Overloads ---
  constexpr uint128_t operator>>(const uint128_t& shift) const
  {
    return *this >> static_cast<int>(shift.low);
  }

  constexpr uint128_t operator<<(const uint128_t& shift) const
  {
    return *this << static_cast<int>(shift.low);
  }

  constexpr uint128_t& operator<<=(int shift)
  {
    *this = *this << shift;
    return *this;
  }
  constexpr uint128_t& operator>>=(int shift)
  {
    *this = *this >> shift;
    return *this;
  }

  constexpr uint128_t& operator<<=(const uint128_t& shift)
  {
    *this = *this << shift;
    return *this;
  }
  constexpr uint128_t& operator>>=(const uint128_t& shift)
  {
    *this = *this >> shift;
    return *this;
  }

  constexpr uint128_t operator|(const uint128_t& rhs) const
  {
    return uint128_t(low | rhs.low, high | rhs.high);
  }
  constexpr uint128_t operator&(const uint128_t& rhs) const
  {
    return uint128_t(low & rhs.low, high & rhs.high);
  }
  constexpr uint128_t operator^(const uint128_t& rhs) const
  {
    return uint128_t(low ^ rhs.low, high ^ rhs.high);
  }
  constexpr uint128_t operator~() const { return uint128_t(~low, ~high); }

  // --- Comparison Operators ---
  constexpr bool operator==(const uint128_t& rhs) const
  {
    return low == rhs.low && high == rhs.high;
  }
  constexpr bool operator!=(const uint128_t& rhs) const
  {
    return !(*this == rhs);
  }
  constexpr bool operator<(const uint128_t& rhs) const
  {
    return high < rhs.high || (high == rhs.high && low < rhs.low);
  }
  constexpr bool operator>(const uint128_t& rhs) const { return rhs < *this; }
  constexpr bool operator<=(const uint128_t& rhs) const
  {
    return !(*this > rhs);
  }
  constexpr bool operator>=(const uint128_t& rhs) const
  {
    return !(*this < rhs);
  }
};
