// Pure-AVX2 ternary (int2) x int8 microkernels -- a drop-in replacement for the
// LIBXSMM INTERPRETE_A_AS_INT2_VNNI4_INTLV BRGEMM, removing the dependency.
//
// Reproduces libxsmm_generator_gemm_avx2_microkernel_int2int8_srf: one 32-byte
// ymm load of packed weights is expanded to four int8 registers via two
// VPSHUFB lookups on each nibble, then accumulated with VPDPBSSD.
//
//   a0 = lut01[b & 0xf]        -> k3=0, output rows  0..7
//   a1 = lut23[b & 0xf]        -> k3=1, output rows  8..15
//   a2 = lut01[(b >> 4) & 0xf] -> k3=2, output rows 16..23
//   a3 = lut23[(b >> 4) & 0xf] -> k3=3, output rows 24..31
//
// Within each register the 8 int32 lanes are k2, so accumulator `m` lane `L`
// holds output row m*8 + L, matching the packer's r = k3*8 + k2.
//
// Register budget (16 ymm): 3 constants + 4 expanded A + 1 broadcast B leaves 8
// for accumulators, so at most 2 activation rows fit without spilling.

#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define XSMM_TERNARY_HAVE_X86 1
#endif

namespace vllm_ternary {

// 32 output columns per tile, VNNI4 along the reduction dim.
inline constexpr int kMicroN = 32;
inline constexpr int kMicroVnni = 4;
// activation rows per microkernel call before accumulators spill
inline constexpr int kMicroMaxRows = 2;

#ifdef XSMM_TERNARY_HAVE_X86

// Decode tables, identical to LIBXSMM's perm_bits_01 / perm_bits_23 / mask_bits.
// Code 3 never occurs (the packer emits only 0, 1, 2) so those slots are zero.
struct TernaryLuts {
  __m256i lut01;
  __m256i lut23;
  __m256i mask;
};

__attribute__((target("avx2"))) inline TernaryLuts make_ternary_luts() {
  alignas(32) int8_t p01[32];
  alignas(32) int8_t p23[32];
  for (int i = 0; i < 32; ++i) {
    const int lo = i & 0x3;
    p01[i] = (lo == 1) ? 1 : (lo == 2 ? -1 : 0);
    const int hi = i & 0xc;
    p23[i] = (hi == 4) ? 1 : (hi == 8 ? -1 : 0);
  }
  TernaryLuts l;
  l.lut01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p01));
  l.lut23 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p23));
  l.mask = _mm256_set1_epi8(0x0f);
  return l;
}

// KFIXED > 0 makes the trip count a compile-time constant so GCC drops the
// remainder loop. The unroll factor differs by NROWS: with 8 accumulators live
// (NROWS=2) an unroll of 4 exhausts the 16 ymm and GCC spills every iteration,
// so that case unrolls by 2.
template <int NROWS, int KFIXED>
__attribute__((target("avx2,avxvnniint8"))) inline void ternary_micro_impl(
    const uint8_t* __restrict a,
    const int8_t* __restrict b,
    int64_t ldb,
    int32_t* __restrict c,
    int64_t K_rt,
    const TernaryLuts& luts) {
  static_assert(NROWS >= 1 && NROWS <= kMicroMaxRows, "accumulators would spill");
  const int64_t K = (KFIXED > 0) ? static_cast<int64_t>(KFIXED) : K_rt;

  const __m256i lut01 = luts.lut01;
  const __m256i lut23 = luts.lut23;
  const __m256i mask = luts.mask;

  __m256i c00 = _mm256_setzero_si256();
  __m256i c01 = _mm256_setzero_si256();
  __m256i c02 = _mm256_setzero_si256();
  __m256i c03 = _mm256_setzero_si256();
  __m256i c10 = _mm256_setzero_si256();
  __m256i c11 = _mm256_setzero_si256();
  __m256i c12 = _mm256_setzero_si256();
  __m256i c13 = _mm256_setzero_si256();

  const int8_t* __restrict b0 = b;
  const int8_t* __restrict b1 = b + (NROWS > 1 ? ldb : 0);

#define XSMM_TERNARY_STEP()                                              \
  do {                                                                   \
    const __m256i raw =                                                  \
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));         \
    const __m256i lo = _mm256_and_si256(raw, mask);                      \
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi32(raw, 4), mask); \
    const __m256i a0 = _mm256_shuffle_epi8(lut01, lo);                   \
    const __m256i a1 = _mm256_shuffle_epi8(lut23, lo);                   \
    const __m256i a2 = _mm256_shuffle_epi8(lut01, hi);                   \
    const __m256i a3 = _mm256_shuffle_epi8(lut23, hi);                   \
    a += kMicroN;                                                        \
    int32_t p0;                                                          \
    __builtin_memcpy(&p0, b0 + k, sizeof(p0));                           \
    const __m256i bv0 = _mm256_set1_epi32(p0);                           \
    c00 = _mm256_dpbssd_epi32(c00, a0, bv0);                             \
    c01 = _mm256_dpbssd_epi32(c01, a1, bv0);                             \
    c02 = _mm256_dpbssd_epi32(c02, a2, bv0);                             \
    c03 = _mm256_dpbssd_epi32(c03, a3, bv0);                             \
    if constexpr (NROWS > 1) {                                           \
      int32_t p1;                                                        \
      __builtin_memcpy(&p1, b1 + k, sizeof(p1));                         \
      const __m256i bv1 = _mm256_set1_epi32(p1);                         \
      c10 = _mm256_dpbssd_epi32(c10, a0, bv1);                           \
      c11 = _mm256_dpbssd_epi32(c11, a1, bv1);                           \
      c12 = _mm256_dpbssd_epi32(c12, a2, bv1);                           \
      c13 = _mm256_dpbssd_epi32(c13, a3, bv1);                           \
    }                                                                    \
  } while (0)

  if constexpr (NROWS == 1) {
#pragma GCC unroll 4
    for (int64_t k = 0; k < K; k += kMicroVnni) {
      XSMM_TERNARY_STEP();
    }
  } else {
#pragma GCC unroll 4
    for (int64_t k = 0; k < K; k += kMicroVnni) {
      XSMM_TERNARY_STEP();
    }
  }
#undef XSMM_TERNARY_STEP

  _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + 0), c00);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + 8), c01);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + 16), c02);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + 24), c03);
  if constexpr (NROWS > 1) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + kMicroN + 0), c10);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + kMicroN + 8), c11);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + kMicroN + 16), c12);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + kMicroN + 24), c13);
  }
}

template <int NROWS>
inline void ternary_micro(
    const uint8_t* __restrict a,
    const int8_t* __restrict b,
    int64_t ldb,
    int32_t* __restrict c,
    int64_t K,
    const TernaryLuts& luts) {
  // K stays runtime: a compile-time trip count makes GCC spill the accumulators.
  ternary_micro_impl<NROWS, 0>(a, b, ldb, c, K, luts);
}

#endif  // XSMM_TERNARY_HAVE_X86

// Portable reference, used when AVX-VNNI-INT8 is unavailable.
inline void ternary_micro_scalar(
    const uint8_t* __restrict a,
    const int8_t* __restrict b,
    int64_t ldb,
    int32_t* __restrict c,
    int64_t K,
    int nrows) {
  for (int n = 0; n < nrows; ++n) {
    for (int r = 0; r < kMicroN; ++r) {
      c[n * kMicroN + r] = 0;
    }
  }
  for (int64_t k = 0; k < K; k += kMicroVnni) {
    const uint8_t* tile = a + (k / kMicroVnni) * kMicroN;
    for (int k2 = 0; k2 < 8; ++k2) {
      for (int c3 = 0; c3 < kMicroVnni; ++c3) {
        const uint8_t byte = tile[k2 * kMicroVnni + c3];
        for (int k3 = 0; k3 < 4; ++k3) {
          const int code = (byte >> (2 * k3)) & 0x3;
          const int w = (code == 1) ? 1 : (code == 2 ? -1 : 0);
          if (w == 0) continue;
          for (int n = 0; n < nrows; ++n) {
            c[n * kMicroN + k3 * 8 + k2] +=
                w * static_cast<int32_t>(b[n * ldb + k + c3]);
          }
        }
      }
    }
  }
}

bool ternary_intrinsics_supported();

}  // namespace vllm_ternary
