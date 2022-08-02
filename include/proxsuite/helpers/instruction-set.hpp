//
// Copyright (c) 2022 INRIA
//
/**
 * @file instruction-set.hpp
 */

#ifndef PROXSUITE_HELPERS_INSTRUCTION_SET_HPP
#define PROXSUITE_HELPERS_INSTRUCTION_SET_HPP

#include <vector>
#include <bitset>
#include <array>

#ifndef _WIN32
#include <cpuid.h>
#else
#include <intrin.h>
#endif

namespace proxsuite {
namespace helpers {

namespace internal {
inline void
cpuid(std::array<int, 4>& cpui, int level)
{
#ifndef _WIN32
  __cpuid(level, cpui[0], cpui[1], cpui[2], cpui[3]);
#else
  __cpuid(cpui.data(), level);
#endif
}

inline void
cpuidex(std::array<int, 4>& cpui, int level, int count)
{
#ifndef _WIN32
  __cpuid_count(level, count, cpui[0], cpui[1], cpui[2], cpui[3]);
#else
  __cpuidex(cpui.data(), level, count);
#endif
}
}

// Adapted from
// https://docs.microsoft.com/fr-fr/cpp/intrinsics/cpuid-cpuidex?view=msvc-170
struct InstructionSet
{
  static std::string vendor(void) { return data.vendor_; }
  static std::string brand(void) { return data.brand_; }

  static bool has_SSE3(void) { return data.f_1_ECX_[0]; }
  static bool has_PCLMULQDQ(void) { return data.f_1_ECX_[1]; }
  static bool has_MONITOR(void) { return data.f_1_ECX_[3]; }
  static bool has_SSSE3(void) { return data.f_1_ECX_[9]; }
  static bool has_FMA(void) { return data.f_1_ECX_[12]; }
  static bool has_CMPXCHG16B(void) { return data.f_1_ECX_[13]; }
  static bool has_SSE41(void) { return data.f_1_ECX_[19]; }
  static bool has_SSE42(void) { return data.f_1_ECX_[20]; }
  static bool has_MOVBE(void) { return data.f_1_ECX_[22]; }
  static bool has_POPCNT(void) { return data.f_1_ECX_[23]; }
  static bool has_AES(void) { return data.f_1_ECX_[25]; }
  static bool has_XSAVE(void) { return data.f_1_ECX_[26]; }
  static bool has_OSXSAVE(void) { return data.f_1_ECX_[27]; }
  static bool has_AVX(void) { return data.f_1_ECX_[28]; }
  static bool has_F16C(void) { return data.f_1_ECX_[29]; }
  static bool has_RDRAND(void) { return data.f_1_ECX_[30]; }

  static bool has_MSR(void) { return data.f_1_EDX_[5]; }
  static bool has_CX8(void) { return data.f_1_EDX_[8]; }
  static bool has_SEP(void) { return data.f_1_EDX_[11]; }
  static bool has_CMOV(void) { return data.f_1_EDX_[15]; }
  static bool has_CLFSH(void) { return data.f_1_EDX_[19]; }
  static bool has_MMX(void) { return data.f_1_EDX_[23]; }
  static bool has_FXSR(void) { return data.f_1_EDX_[24]; }
  static bool has_SSE(void) { return data.f_1_EDX_[25]; }
  static bool has_SSE2(void) { return data.f_1_EDX_[26]; }

  static bool has_FSGSBASE(void) { return data.f_7_EBX_[0]; }
  static bool has_AVX512VBMI(void) { return data.f_7_EBX_[1]; }
  static bool has_BMI1(void) { return data.f_7_EBX_[3]; }
  static bool has_HLE(void) { return data.isIntel_ && data.f_7_EBX_[4]; }
  static bool has_AVX2(void) { return data.f_7_EBX_[5]; }
  static bool has_BMI2(void) { return data.f_7_EBX_[8]; }
  static bool has_ERMS(void) { return data.f_7_EBX_[9]; }
  static bool has_INVPCID(void) { return data.f_7_EBX_[10]; }
  static bool has_RTM(void) { return data.isIntel_ && data.f_7_EBX_[11]; }
  static bool has_AVX512F(void) { return data.f_7_EBX_[16]; }
  static bool has_AVX512DQ(void) { return data.f_7_EBX_[17]; }
  static bool has_RDSEED(void) { return data.f_7_EBX_[18]; }
  static bool has_ADX(void) { return data.f_7_EBX_[19]; }
  static bool has_AVX512IFMA(void) { return data.f_7_EBX_[21]; }
  static bool has_AVX512PF(void) { return data.f_7_EBX_[26]; }
  static bool has_AVX512ER(void) { return data.f_7_EBX_[27]; }
  static bool has_AVX512CD(void) { return data.f_7_EBX_[28]; }
  static bool has_SHA(void) { return data.f_7_EBX_[29]; }
  static bool has_AVX512BW(void) { return data.f_7_EBX_[30]; }
  static bool has_AVX512VL(void) { return data.f_7_EBX_[31]; }

  static bool has_PREFETCHWT1(void) { return data.f_7_ECX_[0]; }

  static bool has_LAHF(void) { return data.f_81_ECX_[0]; }
  static bool has_LZCNT(void) { return data.isIntel_ && data.f_81_ECX_[5]; }
  static bool has_ABM(void) { return data.isAMD_ && data.f_81_ECX_[5]; }
  static bool has_SSE4a(void) { return data.isAMD_ && data.f_81_ECX_[6]; }
  static bool has_XOP(void) { return data.isAMD_ && data.f_81_ECX_[11]; }
  static bool has_FMA4(void) { return data.isAMD_ && data.f_81_ECX_[16]; }
  static bool has_TBM(void) { return data.isAMD_ && data.f_81_ECX_[21]; }

  static bool has_SYSCALL(void) { return data.isIntel_ && data.f_81_EDX_[11]; }
  static bool has_MMXEXT(void) { return data.isAMD_ && data.f_81_EDX_[22]; }
  static bool has_RDTSCP(void) { return data.isIntel_ && data.f_81_EDX_[27]; }
  static bool has_x64(void) { return data.isIntel_ && data.f_81_EDX_[29]; }
  static bool has_3DNOWEXT(void) { return data.isAMD_ && data.f_81_EDX_[30]; }
  static bool has_3DNOW(void) { return data.isAMD_ && data.f_81_EDX_[31]; }

private:
  struct Data
  {
    Data()
      : nIds_{ 0 }
      , nExIds_{ 0 }
      , isIntel_{ false }
      , isAMD_{ false }
      , f_1_ECX_{ 0 }
      , f_1_EDX_{ 0 }
      , f_7_EBX_{ 0 }
      , f_7_ECX_{ 0 }
      , f_81_ECX_{ 0 }
      , f_81_EDX_{ 0 }
      , data_{}
      , extdata_{}
    {
      std::array<int, 4> cpui;
      typedef unsigned long long bistset_equivalent_type;

      // Calling __cpuid with 0x0 as the function_id argument
      // gets the number of the highest valid function ID.
      internal::cpuid(cpui, 0);
      nIds_ = cpui[0];

      for (int i = 0; i <= nIds_; ++i) {
        internal::cpuidex(cpui, i, 0);
        data_.push_back(cpui);
      }

      // Capture vendor string
      char vendor[0x20];
      memset(vendor, 0, sizeof(vendor));
      *reinterpret_cast<int*>(vendor) = data_[0][1];
      *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
      *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
      vendor_ = vendor;
      if (vendor_ == "GenuineIntel") {
        isIntel_ = true;
      } else if (vendor_ == "AuthenticAMD") {
        isAMD_ = true;
      }

      // load bitset with flags for function 0x00000001
      if (nIds_ >= 1) {
        f_1_ECX_ = static_cast<bistset_equivalent_type>(data_[1][2]);
        f_1_EDX_ = static_cast<bistset_equivalent_type>(data_[1][3]);
      }

      // load bitset with flags for function 0x00000007
      if (nIds_ >= 7) {
        f_7_EBX_ = static_cast<bistset_equivalent_type>(data_[7][1]);
        f_7_ECX_ = static_cast<bistset_equivalent_type>(data_[7][2]);
      }

      // Calling __cpuid with 0x80000000 as the function_id argument
      // gets the number of the highest valid extended ID.
      internal::cpuid(cpui, static_cast<int>(0x80000000));
      nExIds_ = cpui[0];

      char brand[0x40];
      memset(brand, 0, sizeof(brand));

      for (int i = static_cast<int>(0x80000000); i <= nExIds_; ++i) {
        internal::cpuidex(cpui, i, 0);
        extdata_.push_back(cpui);
      }

      // load bitset with flags for function 0x80000001
      if (nExIds_ >= static_cast<int>(0x80000001)) {
        f_81_ECX_ = static_cast<bistset_equivalent_type>(extdata_[1][2]);
        f_81_EDX_ = static_cast<bistset_equivalent_type>(extdata_[1][3]);
      }

      // Interpret CPU brand string if reported
      if (nExIds_ >= static_cast<int>(0x80000004)) {
        memcpy(brand, extdata_[2].data(), sizeof(cpui));
        memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
        memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
        brand_ = brand;
      }
    };

    int nIds_;
    int nExIds_;
    std::string vendor_;
    std::string brand_;
    bool isIntel_;
    bool isAMD_;
    std::bitset<32> f_1_ECX_;
    std::bitset<32> f_1_EDX_;
    std::bitset<32> f_7_EBX_;
    std::bitset<32> f_7_ECX_;
    std::bitset<32> f_81_ECX_;
    std::bitset<32> f_81_EDX_;
    std::vector<std::array<int, 4>> data_;
    std::vector<std::array<int, 4>> extdata_;
  };

  inline static const Data data = Data();
};

} // helpers
} // proxsuite

#endif // ifndef PROXSUITE_HELPERS_INSTRUCTION_SET_HPP
