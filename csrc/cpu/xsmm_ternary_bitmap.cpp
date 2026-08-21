// Bitmap + compacted-sign ternary GEMV, ported from the standalone kernel in
// new_ternary_pack_opt.cpp.
//
// The weights arrive in the int2 VNNI4 layout that the LIBXSMM path uses, so
// they are repacked once at load time into
//   bitmap  uint32 [N/32][C]        bit j = "row n1*32+j is non-zero at c"
//   signs   uint32 bit stream       signs of the non-zeros in (c, lane) order,
//                                   each block padded to a whole 32-bit word
//   sgn_off int64  [N/32]           bit offset of each block's run
//   scales  fp16   [N/32][C/G][32]
// and the checkpoint format is left alone.
//
// Reconstruction is three instructions per 32 weights: pdep scatters the
// compacted sign bits onto the present lanes, then two masked moves select
// between the hoisted +scale and -scale vectors.
//
// Needs AVX512-FP16 (fp16 FMA) and BMI2 (pdep); this file is compiled with the
// extra flags and every entry point is guarded by xsmm_ternary_bitmap_supported().

#include <torch/all.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

// defined in xsmm_ternary.cpp; lets both kernels share one set of counters
void xsmm_ternary_record(int64_t rows, int64_t bytes, int64_t nanos);

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// AVX-512 is requested per function, not per file: the file also contains the
// scalar pack loops and the AVX2 kernel, which must stay executable on hosts
// without AVX-512.
#if defined(__x86_64__)
#define XSMM_BITMAP_ISA 1
#define XSMM_AVX512_FN \
  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512fp16,bmi2")))
#else
#define XSMM_AVX512_FN
#endif

#if defined(__AVX2__)
#define XSMM_BITMAP_AVX2 1
#endif

// Arrow Lake has no AVX-512, so it takes the vpdpbssd path with its own layout.
static bool has_avx512_path() {
#ifdef XSMM_BITMAP_ISA
  return __builtin_cpu_supports("avx512fp16") && __builtin_cpu_supports("bmi2");
#else
  return false;
#endif
}
static bool has_avx2_path() {
#ifdef XSMM_BITMAP_AVX2
  return __builtin_cpu_supports("avxvnniint8") && __builtin_cpu_supports("bmi2");
#else
  return false;
#endif
}
static bool use_avx2_path() { return !has_avx512_path() && has_avx2_path(); }

namespace {

constexpr int64_t kBlockN = 32;  // output rows per bitmap block
constexpr int64_t kVnni = 4;
constexpr int64_t kLanes = 8;
constexpr int64_t kPack = 4;  // int2 codes per packed byte

// Independent fp16 accumulators; breaks the serial fmadd latency chain.
#ifndef XSMM_BITMAP_NACC
#define XSMM_BITMAP_NACC 4
#endif
constexpr int kNacc = XSMM_BITMAP_NACC;

// GCC ignores the bare `unroll` pragma, and vLLM compiles at -O2 where its own
// unroller is off; without this any accumulator array is spilled to the stack
// and reloaded every iteration.
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#define XSMM_UNROLL _Pragma("unroll")
#else
#define XSMM_UNROLL _Pragma("GCC unroll 8")
#endif

// Prefetch distances in k iterations, tuned on Emerald Rapids.
constexpr int kPfBmp = 256;
constexpr int kPfSgn = 128;

// int2 code -> ternary. 0 -> 0, 1 -> +1, 2 -> -1.
inline int decode_code(const uint8_t* tile, int64_t C2, int64_t c_local,
                       int64_t row) {
  const int64_t c2 = c_local / kVnni, c3 = c_local % kVnni;
  const int64_t k3 = row / kLanes, k2 = row % kLanes;
  const uint8_t byte = tile[(c2 * kLanes + k2) * kVnni + c3];
  const int code = (byte >> (2 * k3)) & 3;
  return (code == 1) ? 1 : ((code == 2) ? -1 : 0);
}

#ifdef XSMM_BITMAP_ISA

// One 32-row block: y[0..31] = sum_c W[block][c] * x[c], fp16 accumulation.
XSMM_AVX512_FN
static void gemv_block(const uint32_t* __restrict bmp,
                              const uint32_t* __restrict sgn, uint64_t bitpos,
                              const uint16_t* __restrict sc,
                              const _Float16* __restrict x, int64_t C,
                              int64_t G, _Float16* __restrict y) {
  const int64_t n_groups = C / G;
  constexpr int pfs = kPfSgn * 4;
  int64_t k = 0;

  __m512h accv[kNacc];
XSMM_UNROLL
  for (int u = 0; u < kNacc; ++u) accv[u] = _mm512_setzero_ph();

  for (int64_t g = 0; g < n_groups; ++g) {
    const __m512i spos =
        _mm512_loadu_si512((const void*)(sc + (size_t)g * kBlockN));
    const __m512i sneg = _mm512_or_si512(spos, _mm512_set1_epi16((short)0x8000));

    for (int64_t kk = 0; kk < G; kk += kNacc) {
XSMM_UNROLL
      for (int u = 0; u < kNacc; ++u, ++k) {
        const uint32_t m1 = bmp[k];

        uint64_t win;
        const char* sp = (const char*)sgn + ((bitpos >> 5) << 2);
        std::memcpy(&win, sp, 8);
        const uint32_t s2 = (uint32_t)(win >> (bitpos & 31));

        _mm_prefetch((const char*)(bmp + k + kPfBmp), _MM_HINT_T0);
        _mm_prefetch(sp + pfs, _MM_HINT_T0);

        const uint32_t m3 = (uint32_t)_pdep_u32(s2, m1);
        __m512i w = _mm512_maskz_mov_epi16((__mmask32)m1, spos);
        w = _mm512_mask_mov_epi16(w, (__mmask32)m3, sneg);

        accv[u] = _mm512_fmadd_ph(_mm512_castsi512_ph(w),
                                  _mm512_set1_ph(x[k]), accv[u]);

        bitpos += (uint32_t)_mm_popcnt_u32(m1);
      }
    }
  }

  __m512h acc = accv[0];
XSMM_UNROLL
  for (int u = 1; u < kNacc; ++u) acc = _mm512_add_ph(acc, accv[u]);
  _mm512_storeu_si512((void*)y, _mm512_castph_si512(acc));
}

#endif  // XSMM_BITMAP_ISA

#ifdef XSMM_BITMAP_AVX2

// dtypes for which we have a vectorized 8-wide load
template <typename T>
struct Avx2Loadable : std::false_type {};
template <>
struct Avx2Loadable<float> : std::true_type {};
template <>
struct Avx2Loadable<at::Half> : std::true_type {};
template <>
struct Avx2Loadable<at::BFloat16> : std::true_type {};

inline __m256 load8_ps(const float* p) { return _mm256_loadu_ps(p); }

inline __m256 load8_ps(const at::Half* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

inline __m256 load8_ps(const at::BFloat16* p) {
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

inline float hmax_ps(__m256 v) {
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}

// 32 floats -> 32 saturated int8, undoing the lane interleave the two pack
// stages introduce.
inline __m256i pack32_ps_to_epi8(__m256 a, __m256 b, __m256 c, __m256 d) {
  const __m256i ab =
      _mm256_packs_epi32(_mm256_cvtps_epi32(a), _mm256_cvtps_epi32(b));
  const __m256i cd =
      _mm256_packs_epi32(_mm256_cvtps_epi32(c), _mm256_cvtps_epi32(d));
  const __m256i abcd = _mm256_packs_epi16(ab, cd);
  return _mm256_permutevar8x32_epi32(
      abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

// 32 bits -> 32 bytes, 0xFF where set. AVX2 has no mask registers.
static inline __m256i mask_to_bytes(uint32_t m, __m256i shuf, __m256i bit) {
  __m256i v = _mm256_set1_epi32((int)m);
  v = _mm256_shuffle_epi8(v, shuf);
  v = _mm256_and_si256(v, bit);
  return _mm256_cmpeq_epi8(v, bit);
}

// One 32-row block, int8 x int8 -> int32 via vpdpbssd. 8 lanes per ymm means
// 32 rows need exactly 4 accumulators.
static inline void gemv_block_avx2(const uint32_t* __restrict bmp,
                                   const uint8_t* __restrict sgn,
                                   const int64_t* __restrict off,
                                   const uint16_t* __restrict sc,
                                   const int8_t* __restrict xq,
                                   const float* __restrict xs, int64_t G,
                                   int64_t C1, float* __restrict y) {
  const __m256i shuf = _mm256_setr_epi8(
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
  const __m256i bit = _mm256_setr_epi8(
      1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128,
      1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);

  __m256 out[4];
XSMM_UNROLL
  for (int j = 0; j < 4; ++j) out[j] = _mm256_setzero_ps();

  const int64_t qpg = G / kVnni;

  for (int64_t c1 = 0; c1 < C1; ++c1) {
    __m256i acc[4];
XSMM_UNROLL
    for (int j = 0; j < 4; ++j) acc[j] = _mm256_setzero_si256();

    uint64_t bitpos = (uint64_t)off[c1];
    const uint32_t* bq = bmp + c1 * qpg * 4;
    const int8_t* xp = xq + c1 * G;

    for (int64_t q = 0; q < qpg; ++q) {
      int32_t xw;
      std::memcpy(&xw, xp + q * kVnni, 4);
      const __m256i xb = _mm256_set1_epi32(xw);

XSMM_UNROLL
      for (int j = 0; j < 4; ++j) {
        const uint32_t m1 = bq[q * 4 + j];

        uint64_t win;
        // 32-bit aligned base keeps >= 33 valid bits in the 64-bit window
        std::memcpy(&win, sgn + ((bitpos >> 5) << 2), 8);
        const uint32_t s2 = (uint32_t)(win >> (bitpos & 31));
        const uint32_t m3 = (uint32_t)_pdep_u32(s2, m1);

        const __m256i P = mask_to_bytes(m1, shuf, bit);
        const __m256i Nn = mask_to_bytes(m3, shuf, bit);
        __m256i wv = _mm256_sub_epi8(_mm256_setzero_si256(), P);
        wv = _mm256_add_epi8(wv, _mm256_add_epi8(Nn, Nn));

        acc[j] = _mm256_dpbssd_epi32(acc[j], wv, xb);
        bitpos += (uint32_t)_mm_popcnt_u32(m1);
      }
    }

    const __m256 xsv = _mm256_set1_ps(xs[c1]);
    const uint16_t* wsc = sc + c1 * kBlockN;
XSMM_UNROLL
    for (int j = 0; j < 4; ++j) {
      const __m256 ws = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(wsc + j * 8)));
      const __m256 s = _mm256_mul_ps(xsv, ws);
      out[j] =
          _mm256_add_ps(out[j], _mm256_mul_ps(s, _mm256_cvtepi32_ps(acc[j])));
    }
  }

XSMM_UNROLL
  for (int j = 0; j < 4; ++j) _mm256_storeu_ps(y + j * 8, out[j]);
}

// Symmetric int8 quantization of one group of G activations (G % 32 == 0).
template <typename scalar_t>
static inline void quantize_group_avx2(const scalar_t* __restrict src,
                                       int8_t* __restrict dst, int64_t G,
                                       float& scale_out) {
  if constexpr (Avx2Loadable<scalar_t>::value) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 vmax = _mm256_setzero_ps();
    for (int64_t j = 0; j < G; j += 8)
      vmax = _mm256_max_ps(vmax, _mm256_and_ps(load8_ps(src + j), absmask));
    const float scale = hmax_ps(vmax) / 127.0f;
    scale_out = scale;
    const __m256 inv = _mm256_set1_ps(scale != 0.f ? 1.0f / scale : 0.f);
    for (int64_t j = 0; j < G; j += 32) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(dst + j),
          pack32_ps_to_epi8(_mm256_mul_ps(load8_ps(src + j + 0), inv),
                            _mm256_mul_ps(load8_ps(src + j + 8), inv),
                            _mm256_mul_ps(load8_ps(src + j + 16), inv),
                            _mm256_mul_ps(load8_ps(src + j + 24), inv)));
    }
  } else {
    float amax = 0.f;
    for (int64_t j = 0; j < G; ++j)
      amax = std::max(amax, std::fabs((float)src[j]));
    const float scale = amax / 127.0f;
    const float inv = scale != 0.f ? 1.0f / scale : 0.f;
    scale_out = scale;
    for (int64_t j = 0; j < G; ++j) {
      const float q = std::nearbyintf((float)src[j] * inv);
      dst[j] = (int8_t)std::min(std::max(q, -127.0f), 127.0f);
    }
  }
}

template <typename scalar_t>
void run_bitmap_gemv_avx2(const scalar_t* __restrict xin,
                          scalar_t* __restrict outp,
                          const scalar_t* __restrict bp, int8_t* __restrict xq,
                          float* __restrict xs, float* __restrict ybuf,
                          const uint32_t* bm, const uint8_t* sg,
                          const int64_t* off, const uint16_t* sc, int64_t M,
                          int64_t N, int64_t N1, int64_t C, int64_t G,
                          int64_t C1) {
  const int64_t quads = C / kVnni;

  for (int64_t m = 0; m < M; ++m) {
    // Symmetric per-group int8 quantization of the activation row. This runs
    // serially before the block loop, so at M=1 it is pure added latency.
    for (int64_t c1 = 0; c1 < C1; ++c1) {
      quantize_group_avx2<scalar_t>(xin + m * C + c1 * G, xq + c1 * G, G,
                                    xs[c1]);
    }

#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t n1 = 0; n1 < N1; ++n1) {
      alignas(64) float tmp[kBlockN];
      gemv_block_avx2(bm + n1 * quads * 4, sg, off + n1 * C1,
                      sc + n1 * C1 * kBlockN, xq, xs, G, C1, tmp);
      scalar_t* dst = outp + m * N + n1 * kBlockN;
      if (bp != nullptr) {
        const scalar_t* b = bp + n1 * kBlockN;
        for (int j = 0; j < kBlockN; ++j)
          dst[j] = (scalar_t)(tmp[j] + (float)b[j]);
      } else {
        for (int j = 0; j < kBlockN; ++j) dst[j] = (scalar_t)tmp[j];
      }
    }
  }
}

// Merged gate_up projection with silu(gate)*up folded into the block epilogue:
// the gate and up halves are 32-row blocks n1 and n1 + N1/2, so one work item
// owns both and the [M, 2I] intermediate never reaches memory.
template <typename scalar_t>
void run_bitmap_gemv_silu_mul_avx2(
    const scalar_t* __restrict xin, scalar_t* __restrict outp,
    int8_t* __restrict xq, float* __restrict xs, const uint32_t* bm,
    const uint8_t* sg, const int64_t* off, const uint16_t* sc, int64_t M,
    int64_t N1, int64_t C, int64_t G, int64_t C1) {
  const int64_t quads = C / kVnni;
  const int64_t N1h = N1 / 2;
  const int64_t I = N1h * kBlockN;

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t c1 = 0; c1 < C1; ++c1) {
      quantize_group_avx2<scalar_t>(xin + m * C + c1 * G, xq + c1 * G, G,
                                    xs[c1]);
    }

#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t n1 = 0; n1 < N1h; ++n1) {
      alignas(64) float g[kBlockN];
      alignas(64) float u[kBlockN];
      const int64_t nu = n1 + N1h;
      gemv_block_avx2(bm + n1 * quads * 4, sg, off + n1 * C1,
                      sc + n1 * C1 * kBlockN, xq, xs, G, C1, g);
      gemv_block_avx2(bm + nu * quads * 4, sg, off + nu * C1,
                      sc + nu * C1 * kBlockN, xq, xs, G, C1, u);
      scalar_t* dst = outp + m * I + n1 * kBlockN;
      for (int j = 0; j < kBlockN; ++j) {
        dst[j] = (scalar_t)(g[j] / (1.0f + std::exp(-g[j])) * u[j]);
      }
    }
  }
}

// Fused ternary MoE over the bitmap layout.
//
// Every (token, slot) pair is one GEMV, so unlike the int2 path there is no
// counting sort: grouping tokens by expert only pays off for a real GEMM.
// The work is flattened into (pair, 32-row block) items instead, so each of
// the two GEMVs costs one parallel region per layer rather than one per expert.
template <typename scalar_t>
void run_bitmap_moe_avx2(
    const scalar_t* __restrict xin, scalar_t* __restrict outp,
    const int32_t* __restrict ids, const float* __restrict wts,
    const uint32_t* w13_bm, const uint8_t* w13_sg, const int64_t* w13_off,
    const uint16_t* w13_sc, const uint32_t* w2_bm, const uint8_t* w2_sg,
    const int64_t* w2_off, const uint16_t* w2_sc, int8_t* __restrict xq,
    float* __restrict xs, float* __restrict hbuf, int8_t* __restrict aq,
    float* __restrict as, float* __restrict ybuf, int64_t T, int64_t K,
    int64_t H, int64_t I, int64_t G) {
  const int64_t N13 = 2 * I;
  const int64_t C1 = H / G;
  const int64_t D1 = I / G;
  const int64_t n13_blocks = N13 / kBlockN;
  const int64_t h_blocks = H / kBlockN;
  const int64_t h_quads = H / kVnni;
  const int64_t i_quads = I / kVnni;
  const int64_t TK = T * K;

  // one int8 activation row per token, shared by that token's K experts
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c1 = 0; c1 < C1; ++c1) {
      quantize_group_avx2<scalar_t>(xin + t * H + c1 * G, xq + t * H + c1 * G,
                                    G, xs[t * C1 + c1]);
    }
  }

#pragma omp parallel
  {
#ifdef __AVX512F__
#pragma omp for schedule(static)
#else
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t w = 0; w < TK * n13_blocks; ++w) {
      const int64_t p = w / n13_blocks, b = w % n13_blocks;
      const int64_t blk = (int64_t)ids[p] * n13_blocks + b;
      gemv_block_avx2(w13_bm + blk * h_quads * 4, w13_sg, w13_off + blk * C1,
                      w13_sc + blk * C1 * kBlockN, xq + (p / K) * H,
                      xs + (p / K) * C1, G, C1, hbuf + p * N13 + b * kBlockN);
    }

    // SiLU-gated activation in place over the gate half, then requantize
#pragma omp for schedule(static)
    for (int64_t p = 0; p < TK; ++p) {
      float* hp = hbuf + p * N13;
      for (int64_t c = 0; c < I; ++c) {
        const float g = hp[c];
        hp[c] = (g / (1.0f + std::exp(-g))) * hp[I + c];
      }
      for (int64_t c = 0; c < D1; ++c) {
        quantize_group_avx2<float>(hp + c * G, aq + p * I + c * G, G,
                                   as[p * D1 + c]);
      }
    }

#ifdef __AVX512F__
#pragma omp for schedule(static)
#else
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t w = 0; w < TK * h_blocks; ++w) {
      const int64_t p = w / h_blocks, b = w % h_blocks;
      const int64_t blk = (int64_t)ids[p] * h_blocks + b;
      gemv_block_avx2(w2_bm + blk * i_quads * 4, w2_sg, w2_off + blk * D1,
                      w2_sc + blk * D1 * kBlockN, aq + p * I, as + p * D1, G,
                      D1, ybuf + p * H + b * kBlockN);
    }

    // each token owns a disjoint output row, so no atomics are needed
#pragma omp for schedule(static)
    for (int64_t t = 0; t < T; ++t) {
      scalar_t* o = outp + t * H;
      for (int64_t c = 0; c < H; ++c) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += ybuf[(t * K + k) * H + c] * wts[t * K + k];
        }
        o[c] = (scalar_t)sum;
      }
    }
  }
}

#endif  // XSMM_BITMAP_AVX2

#ifdef XSMM_BITMAP_ISA
// OpenMP loops cannot live inside an AT_DISPATCH macro argument, so the whole
// typed body sits here and the lambda just calls it.
template <typename scalar_t>
XSMM_AVX512_FN void run_bitmap_gemv(const scalar_t* __restrict xin, scalar_t* __restrict outp,
                     const scalar_t* __restrict bp, _Float16* __restrict xp,
                     const uint32_t* bm, const uint32_t* sg, const int64_t* off,
                     const uint16_t* sc, int64_t M, int64_t N, int64_t N1,
                     int64_t C, int64_t G, int64_t sc_stride) {
  for (int64_t i = 0; i < M * C; ++i) xp[i] = (_Float16)(float)xin[i];

  for (int64_t m = 0; m < M; ++m) {
#pragma omp parallel for schedule(static)
    for (int64_t n1 = 0; n1 < N1; ++n1) {
      alignas(64) _Float16 tmp[kBlockN];
      gemv_block(bm + n1 * C, sg, (uint64_t)off[n1], sc + n1 * sc_stride,
                 xp + m * C, C, G, tmp);
      scalar_t* dst = outp + m * N + n1 * kBlockN;
      if (bp != nullptr) {
        const scalar_t* b = bp + n1 * kBlockN;
        for (int j = 0; j < kBlockN; ++j) {
          dst[j] = (scalar_t)((float)tmp[j] + (float)b[j]);
        }
      } else {
        for (int j = 0; j < kBlockN; ++j) dst[j] = (scalar_t)(float)tmp[j];
      }
    }
  }
}
template <typename scalar_t>
XSMM_AVX512_FN void run_bitmap_gemv_silu_mul(
    const scalar_t* __restrict xin, scalar_t* __restrict outp,
    _Float16* __restrict xp, const uint32_t* bm, const uint32_t* sg,
    const int64_t* off, const uint16_t* sc, int64_t M, int64_t N1, int64_t C,
    int64_t G, int64_t sc_stride) {
  for (int64_t i = 0; i < M * C; ++i) xp[i] = (_Float16)(float)xin[i];

  const int64_t N1h = N1 / 2;
  const int64_t I = N1h * kBlockN;
  for (int64_t m = 0; m < M; ++m) {
#pragma omp parallel for schedule(static)
    for (int64_t n1 = 0; n1 < N1h; ++n1) {
      alignas(64) _Float16 g[kBlockN];
      alignas(64) _Float16 u[kBlockN];
      const int64_t nu = n1 + N1h;
      gemv_block(bm + n1 * C, sg, (uint64_t)off[n1], sc + n1 * sc_stride,
                 xp + m * C, C, G, g);
      gemv_block(bm + nu * C, sg, (uint64_t)off[nu], sc + nu * sc_stride,
                 xp + m * C, C, G, u);
      scalar_t* dst = outp + m * I + n1 * kBlockN;
      for (int j = 0; j < kBlockN; ++j) {
        const float gv = (float)g[j];
        dst[j] = (scalar_t)(gv / (1.0f + std::exp(-gv)) * (float)u[j]);
      }
    }
  }
}
#endif

}  // namespace

bool xsmm_ternary_bitmap_supported() {
  return has_avx512_path() || has_avx2_path();
}

// AVX2 layout: bitmap [N/32][C/4][4] uint32, bit b of mask j is
// (row 8j + b/4, column quad*4 + b%4); one sign offset per (block, group).
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
bitmap_pack_avx2(const torch::Tensor& packed, const torch::Tensor& scales,
                 int64_t group_size) {
  const int64_t N1 = packed.size(0);
  const int64_t C1 = packed.size(1);
  const int64_t G = group_size;
  const int64_t C2 = G / kVnni;
  const int64_t C = C1 * G;
  const int64_t quads = C / kVnni;
  const int64_t qpg = G / kVnni;
  const int64_t tile_bytes = C2 * kLanes * kVnni;
  const int64_t ntiles = N1 * C1;

  auto packed_c = packed.contiguous();
  const uint8_t* p = packed_c.data_ptr<uint8_t>();

  auto bitmap = torch::empty({N1, quads, 4},
                             torch::TensorOptions().dtype(torch::kInt32));
  auto sgn_off =
      torch::empty({ntiles}, torch::TensorOptions().dtype(torch::kInt64));
  uint32_t* bm = reinterpret_cast<uint32_t*>(bitmap.data_ptr<int32_t>());
  int64_t* off = sgn_off.data_ptr<int64_t>();

  std::vector<int64_t> nbits(ntiles, 0);
#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t t = 0; t < ntiles; ++t) {
    const int64_t n1 = t / C1, c1 = t % C1;
    const uint8_t* tile = p + (n1 * C1 + c1) * tile_bytes;
    int64_t cnt = 0;
    for (int64_t q = 0; q < qpg; ++q) {
      for (int64_t j = 0; j < 4; ++j) {
        uint32_t m = 0;
        for (int64_t b = 0; b < 32; ++b) {
          const int64_t r = j * 8 + b / kVnni;
          const int64_t c_local = q * kVnni + (b % kVnni);
          if (decode_code(tile, C2, c_local, r) != 0) m |= 1u << b;
        }
        bm[n1 * quads * 4 + (c1 * qpg + q) * 4 + j] = m;
        cnt += __builtin_popcount(m);
      }
    }
    nbits[t] = (cnt + 7) & ~int64_t(7);  // byte-align so pass 2 is race-free
  }

  int64_t total = 0;
  for (int64_t t = 0; t < ntiles; ++t) {
    off[t] = total;
    total += nbits[t];
  }
  auto signs = torch::zeros({total / 8 + 16},
                            torch::TensorOptions().dtype(torch::kUInt8));
  uint8_t* sg = signs.data_ptr<uint8_t>();

#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t t = 0; t < ntiles; ++t) {
    const int64_t n1 = t / C1, c1 = t % C1;
    const uint8_t* tile = p + (n1 * C1 + c1) * tile_bytes;
    int64_t bp = off[t];
    for (int64_t q = 0; q < qpg; ++q) {
      for (int64_t j = 0; j < 4; ++j) {
        for (int64_t b = 0; b < 32; ++b) {
          const int64_t r = j * 8 + b / kVnni;
          const int64_t c_local = q * kVnni + (b % kVnni);
          const int code = decode_code(tile, C2, c_local, r);
          if (code == 0) continue;
          if (code < 0) sg[bp >> 3] |= (uint8_t)(1u << (bp & 7));
          ++bp;
        }
      }
    }
  }

  return {bitmap, signs, sgn_off, scales.to(torch::kHalf).contiguous()};
}

// packed [N/32, C/G, G/4, 8, 4] uint8 int2 -> (bitmap, signs, sgn_off, scales)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
xsmm_ternary_bitmap_pack(const torch::Tensor& packed,
                         const torch::Tensor& scales, int64_t group_size) {
  TORCH_CHECK(packed.dim() == 5, "packed weight must be 5-D");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");

  const int64_t N1 = packed.size(0);
  const int64_t C1 = packed.size(1);
  const int64_t G = group_size;
  const int64_t C2 = G / kVnni;
  const int64_t C = C1 * G;
  const int64_t tile_bytes = C2 * kLanes * kVnni;

  TORCH_CHECK(packed.size(2) == C2 && packed.size(3) == kLanes &&
                  packed.size(4) == kVnni,
              "packed weight shape does not match group_size ", G);

  if (use_avx2_path()) return bitmap_pack_avx2(packed, scales, group_size);

  auto packed_c = packed.contiguous();
  const uint8_t* p = packed_c.data_ptr<uint8_t>();

  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32);
  auto bitmap = torch::empty({N1, C}, opts_i32);
  auto sgn_off = torch::empty({N1}, torch::TensorOptions().dtype(torch::kInt64));
  uint32_t* bm = reinterpret_cast<uint32_t*>(bitmap.data_ptr<int32_t>());
  int64_t* off = sgn_off.data_ptr<int64_t>();

  std::vector<int64_t> nbits(N1, 0);
#pragma omp parallel for schedule(static)
  for (int64_t n1 = 0; n1 < N1; ++n1) {
    int64_t bits = 0;
    for (int64_t c = 0; c < C; ++c) {
      const int64_t c1 = c / G, c_local = c % G;
      const uint8_t* tile = p + (n1 * C1 + c1) * tile_bytes;
      uint32_t m = 0;
      for (int64_t j = 0; j < kBlockN; ++j) {
        if (decode_code(tile, C2, c_local, j) != 0) m |= 1u << j;
      }
      bm[n1 * C + c] = m;
      bits += __builtin_popcount(m);
    }
    nbits[n1] = (bits + 31) & ~int64_t(31);  // whole 32-bit words per block
  }

  int64_t total = 0;
  for (int64_t n1 = 0; n1 < N1; ++n1) {
    off[n1] = total;
    total += nbits[n1];
  }
  // two extra words so the 64-bit window read at the tail stays in bounds
  auto signs = torch::zeros({total / 32 + 2}, opts_i32);
  uint32_t* sg = reinterpret_cast<uint32_t*>(signs.data_ptr<int32_t>());

  // blocks own disjoint, word-aligned ranges, so this is race-free
#pragma omp parallel for schedule(static)
  for (int64_t n1 = 0; n1 < N1; ++n1) {
    int64_t bp = off[n1];
    for (int64_t c = 0; c < C; ++c) {
      const int64_t c1 = c / G, c_local = c % G;
      const uint8_t* tile = p + (n1 * C1 + c1) * tile_bytes;
      for (int64_t j = 0; j < kBlockN; ++j) {
        const int code = decode_code(tile, C2, c_local, j);
        if (code == 0) continue;
        if (code < 0) sg[bp >> 5] |= 1u << (bp & 31);
        ++bp;
      }
    }
  }

  auto scales_h = scales.to(torch::kHalf).contiguous();
  return {bitmap, signs, sgn_off, scales_h};
}

torch::Tensor xsmm_ternary_bitmap_gemv(
    const torch::Tensor& input, const torch::Tensor& bitmap,
    const torch::Tensor& signs, const torch::Tensor& sgn_off,
    const torch::Tensor& scales, const std::optional<torch::Tensor>& bias,
    int64_t group_size) {
#ifdef XSMM_BITMAP_AVX2
  if (use_avx2_path()) {
    const auto x = input.contiguous();
    const int64_t C = x.size(-1);
    const int64_t M = x.numel() / C;
    const int64_t N1 = bitmap.size(0);
    const int64_t N = N1 * kBlockN;
    const int64_t G = group_size;
    const int64_t C1 = C / G;
    TORCH_CHECK((G & 31) == 0,
                "AVX2 bitmap GEMV needs group_size divisible by 32, got ", G);

    auto sizes = input.sizes().vec();
    sizes.back() = N;
    auto out = torch::empty(sizes, input.options());

    static thread_local std::vector<int8_t> xqbuf;
    static thread_local std::vector<float> xsbuf;
    if ((int64_t)xqbuf.size() < C) xqbuf.resize(C);
    if ((int64_t)xsbuf.size() < C1) xsbuf.resize(C1);

    const uint32_t* bm =
        reinterpret_cast<const uint32_t*>(bitmap.data_ptr<int32_t>());
    const uint8_t* sg = signs.data_ptr<uint8_t>();
    const int64_t* off = sgn_off.data_ptr<int64_t>();
    const uint16_t* sc =
        reinterpret_cast<const uint16_t*>(scales.data_ptr<at::Half>());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "xsmm_bitmap_avx2", [&] {
          run_bitmap_gemv_avx2<scalar_t>(
              x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
              bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
              xqbuf.data(), xsbuf.data(), nullptr, bm, sg, off, sc, M, N, N1, C,
              G, C1);
        });
    return out;
  }
#endif
#ifndef XSMM_BITMAP_ISA
  TORCH_CHECK(false, "xsmm_ternary_bitmap_gemv requires AVX-512");
#else
  const auto x = input.contiguous();
  const int64_t C = x.size(-1);
  const int64_t M = x.numel() / C;
  const int64_t N1 = bitmap.size(0);
  const int64_t N = N1 * kBlockN;
  const int64_t G = group_size;

  TORCH_CHECK(bitmap.size(1) == C, "bitmap width does not match activations");
  TORCH_CHECK(C % G == 0, "C must be a multiple of group_size");

  auto sizes = input.sizes().vec();
  sizes.back() = N;
  auto out = torch::empty(sizes, input.options());

  const uint32_t* bm =
      reinterpret_cast<const uint32_t*>(bitmap.data_ptr<int32_t>());
  const uint32_t* sg =
      reinterpret_cast<const uint32_t*>(signs.data_ptr<int32_t>());
  const int64_t* off = sgn_off.data_ptr<int64_t>();
  const uint16_t* sc =
      reinterpret_cast<const uint16_t*>(scales.data_ptr<at::Half>());
  const int64_t sc_stride = (C / G) * kBlockN;
  const auto t_start = std::chrono::steady_clock::now();

  // The kernel is fp16 throughout. Converting activations and results here
  // rather than with .to()/.copy_() keeps the whole call allocation-free, which
  // matters at ~250 GEMMs per decode step.
  static thread_local std::vector<_Float16> xbuf;
  if ((int64_t)xbuf.size() < M * C) xbuf.resize(M * C);
  _Float16* xp = xbuf.data();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "xsmm_bitmap_gemv", [&] {
        run_bitmap_gemv<scalar_t>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr, xp, bm, sg,
            off, sc, M, N, N1, C, G, sc_stride);
      });

  xsmm_ternary_record(
      M,
      bitmap.numel() * 4 + signs.numel() * 4 + scales.numel() * 2,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_start)
          .count());
  return out;
#endif
}

// Fused ternary MoE. The weights are the per-expert bitmap tensors produced by
// stacking xsmm_ternary_bitmap_pack over the expert axis; sign offsets are
torch::Tensor xsmm_ternary_bitmap_gemv_silu_mul(
    const torch::Tensor& input, const torch::Tensor& bitmap,
    const torch::Tensor& signs, const torch::Tensor& sgn_off,
    const torch::Tensor& scales, int64_t group_size) {
  const auto x = input.contiguous();
  const int64_t C = x.size(-1);
  const int64_t M = x.numel() / C;
  const int64_t N1 = bitmap.size(0);
  const int64_t G = group_size;
  const int64_t C1 = C / G;
  TORCH_CHECK(N1 % 2 == 0, "fused silu_mul needs an even number of row blocks");

  auto sizes = input.sizes().vec();
  sizes.back() = N1 * kBlockN / 2;
  auto out = torch::empty(sizes, input.options());
  const auto t_start = std::chrono::steady_clock::now();

#ifdef XSMM_BITMAP_AVX2
  if (use_avx2_path()) {
    TORCH_CHECK((G & 31) == 0,
                "AVX2 bitmap GEMV needs group_size divisible by 32, got ", G);
    static thread_local std::vector<int8_t> xqbuf;
    static thread_local std::vector<float> xsbuf;
    if ((int64_t)xqbuf.size() < C) xqbuf.resize(C);
    if ((int64_t)xsbuf.size() < C1) xsbuf.resize(C1);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "xsmm_bitmap_silu_avx2",
        [&] {
          run_bitmap_gemv_silu_mul_avx2<scalar_t>(
              x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), xqbuf.data(),
              xsbuf.data(),
              reinterpret_cast<const uint32_t*>(bitmap.data_ptr<int32_t>()),
              signs.data_ptr<uint8_t>(), sgn_off.data_ptr<int64_t>(),
              reinterpret_cast<const uint16_t*>(scales.data_ptr<at::Half>()), M,
              N1, C, G, C1);
        });
    xsmm_ternary_record(M,
                        bitmap.numel() * 4 + signs.numel() +
                            scales.numel() * 2,
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t_start)
                            .count());
    return out;
  }
#endif
#ifndef XSMM_BITMAP_ISA
  TORCH_CHECK(false, "xsmm_ternary_bitmap_gemv_silu_mul requires AVX-512");
#else
  static thread_local std::vector<_Float16> xbuf;
  if ((int64_t)xbuf.size() < M * C) xbuf.resize(M * C);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "xsmm_bitmap_silu", [&] {
        run_bitmap_gemv_silu_mul<scalar_t>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), xbuf.data(),
            reinterpret_cast<const uint32_t*>(bitmap.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(signs.data_ptr<int32_t>()),
            sgn_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint16_t*>(scales.data_ptr<at::Half>()), M,
            N1, C, G, C1 * kBlockN);
      });

  xsmm_ternary_record(
      M, bitmap.numel() * 4 + signs.numel() * 4 + scales.numel() * 2,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_start)
          .count());
  return out;
#endif
}

// Fused ternary MoE. The weights are the per-expert bitmap tensors produced by
// stacking xsmm_ternary_bitmap_pack over the expert axis; sign offsets are
// absolute bit positions into the concatenated sign stream.
torch::Tensor xsmm_ternary_bitmap_moe(
    const torch::Tensor& input, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& w13_bitmap,
    const torch::Tensor& w13_signs, const torch::Tensor& w13_sgn_off,
    const torch::Tensor& w13_scales, const torch::Tensor& w2_bitmap,
    const torch::Tensor& w2_signs, const torch::Tensor& w2_sgn_off,
    const torch::Tensor& w2_scales, int64_t group_size) {
#ifdef XSMM_BITMAP_AVX2
  TORCH_CHECK(use_avx2_path(), "bitmap MoE is only wired for the AVX2 path");
  TORCH_CHECK(input.dim() == 2, "input must be [T, H]");

  const auto t_start = std::chrono::steady_clock::now();

  const auto x = input.contiguous();
  const auto ids = topk_ids.to(torch::kInt).contiguous();
  const auto wts = topk_weights.to(torch::kFloat).contiguous();

  const int64_t T = x.size(0);
  const int64_t H = x.size(1);
  const int64_t K = ids.size(1);
  const int64_t G = group_size;
  const int64_t N13 = w13_bitmap.size(1) * kBlockN;
  const int64_t I = N13 / 2;
  const int64_t TK = T * K;

  TORCH_CHECK((G & 31) == 0, "group_size must be divisible by 32, got ", G);
  TORCH_CHECK(H % G == 0 && I % G == 0,
              "hidden (", H, ") and intermediate (", I,
              ") must be multiples of group_size");
  TORCH_CHECK(w2_bitmap.size(1) * kBlockN == H,
              "w2 output width does not match hidden size");

  auto out = torch::empty({T, H}, x.options());

  static thread_local std::vector<int8_t> xqb, aqb;
  static thread_local std::vector<float> xsb, hb, asb, yb;
  if ((int64_t)xqb.size() < T * H) xqb.resize(T * H);
  if ((int64_t)xsb.size() < T * (H / G)) xsb.resize(T * (H / G));
  if ((int64_t)hb.size() < TK * N13) hb.resize(TK * N13);
  if ((int64_t)aqb.size() < TK * I) aqb.resize(TK * I);
  if ((int64_t)asb.size() < TK * (I / G)) asb.resize(TK * (I / G));
  if ((int64_t)yb.size() < TK * H) yb.resize(TK * H);

  const auto* w13_bm =
      reinterpret_cast<const uint32_t*>(w13_bitmap.data_ptr<int32_t>());
  const auto* w2_bm =
      reinterpret_cast<const uint32_t*>(w2_bitmap.data_ptr<int32_t>());
  const auto* w13_sc =
      reinterpret_cast<const uint16_t*>(w13_scales.data_ptr<at::Half>());
  const auto* w2_sc =
      reinterpret_cast<const uint16_t*>(w2_scales.data_ptr<at::Half>());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "xsmm_bitmap_moe", [&] {
        run_bitmap_moe_avx2<scalar_t>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            ids.data_ptr<int32_t>(), wts.data_ptr<float>(), w13_bm,
            w13_signs.data_ptr<uint8_t>(), w13_sgn_off.data_ptr<int64_t>(),
            w13_sc, w2_bm, w2_signs.data_ptr<uint8_t>(),
            w2_sgn_off.data_ptr<int64_t>(), w2_sc, xqb.data(), xsb.data(),
            hb.data(), aqb.data(), asb.data(), yb.data(), T, K, H, I, G);
      });

  xsmm_ternary_record(
      T,
      (w13_bitmap.numel() + w2_bitmap.numel()) * 4 * K /
          std::max<int64_t>(1, w13_bitmap.size(0)),
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_start)
          .count());
  return out;
#else
  TORCH_CHECK(false, "xsmm_ternary_bitmap_moe requires AVX2");
#endif
}
