// Ternary (2-bit) weight x int8 activation GEMM for x86 CPUs, using LIBXSMM
// JIT microkernels.
//
// Pipeline (mirrors the tpp-pytorch-extension `qint2_intlv` recipe):
//   1. weights are quantized to ternary + per-group fp32 scales and packed into
//      the VNNI4/interleaved layout once, at model load time;
//   2. at inference the 16-bit activations are quantized per (row, group) to
//      int8;
//   3. LIBXSMM runs an int2 x int8 -> int32 BRGEMM;
//   4. the int32 accumulator is dequantized with (act_scale * weight_scale).
//
// The int2 microkernel is only generated for LIBXSMM_X86_AVX2_SRF, i.e. AVX2 +
// AVX-VNNI-INT8 + AVX-NE-CONVERT.

#include <torch/all.h>

#include <libxsmm.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define XSMM_TERNARY_AVX2 1
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define XSMM_TERNARY_AVX512 1
#endif

namespace {

// One 32-byte ymm load of packed weights feeds 32 output columns; fixed by the
// LIBXSMM int2 microkernel.
constexpr int64_t kBlockN = 32;
// VNNI4: 4 consecutive reduction elements are interleaved.
constexpr int64_t kVnni = 4;
// int32 lanes per accumulator register.
constexpr int64_t kLanes = 8;
// ternary values packed per byte, interleaved along N.
constexpr int64_t kPack = 4;
// rows of activations processed per tile.
constexpr int64_t kBlockM = 32;

static_assert(kBlockN == kLanes * kPack, "N block must be lanes x pack factor");

struct GemmKey {
  int32_t m, n, k, lda, ldb, ldc;
  // 2 = int2 interleaved weights, 8 = plain VNNI4 int8 weights
  int32_t wbits;

  bool operator==(const GemmKey& o) const {
    return m == o.m && n == o.n && k == o.k && lda == o.lda && ldb == o.ldb &&
           ldc == o.ldc && wbits == o.wbits;
  }
};

struct GemmKeyHash {
  size_t operator()(const GemmKey& k) const {
    size_t h = 1469598103934665603ULL;
    const auto* p = reinterpret_cast<const uint8_t*>(&k);
    for (size_t i = 0; i < sizeof(GemmKey); ++i) {
      h = (h ^ p[i]) * 1099511628211ULL;
    }
    return h;
  }
};

// JIT kernels are generated once and reused for the lifetime of the process.
libxsmm_gemmfunction get_ternary_kernel(const GemmKey& key) {
  static std::unordered_map<GemmKey, libxsmm_gemmfunction, GemmKeyHash> cache;
  static std::mutex mutex;
  static std::once_flag init_once;
  std::call_once(init_once, [] { libxsmm_init(); });

  std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  libxsmm_gemm_shape shape;
  std::memset(&shape, 0, sizeof(shape));
  // LIBXSMM's A operand is the packed weight, its B operand the activations.
  shape.m = key.m;
  shape.n = key.n;
  shape.k = key.k;
  shape.lda = key.lda;
  shape.ldb = key.ldb;
  shape.ldc = key.ldc;
  shape.a_in_type = LIBXSMM_DATATYPE_I8;
  shape.b_in_type = LIBXSMM_DATATYPE_I8;
  shape.out_type = LIBXSMM_DATATYPE_I32;
  shape.comp_type = LIBXSMM_DATATYPE_I32;

  const libxsmm_bitfield int2_flag =
      (key.wbits == 2) ? LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT2_VNNI4_INTLV : 0;
  const libxsmm_bitfield flags = LIBXSMM_GEMM_FLAGS('N', 'N') |
      LIBXSMM_GEMM_FLAG_VNNI_A | int2_flag | LIBXSMM_GEMM_FLAG_BETA_0;

  libxsmm_gemm_batch_reduce_config brconfig;
  std::memset(&brconfig, 0, sizeof(brconfig));
  brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  // int2 packs 4 ternary values per byte; int8 is one byte per value
  brconfig.br_stride_a_hint =
      (key.wbits == 2) ? key.k * key.lda / kPack : key.k * key.lda;
  brconfig.br_stride_b_hint = key.k;
  brconfig.br_unroll_hint = 1;

  libxsmm_gemmfunction fn =
      libxsmm_dispatch_brgemm(shape, flags, /*prefetch=*/0, brconfig);
  TORCH_CHECK(
      fn != nullptr,
      "LIBXSMM failed to JIT an int",
      key.wbits,
      "xint8 kernel (m=",
      key.m,
      " n=",
      key.n,
      " k=",
      key.k,
      "). The AVX2/SRF microkernel requires AVX-VNNI-INT8 and "
      "AVX-NE-CONVERT; libxsmm target arch is '",
      libxsmm_get_target_arch(),
      "'.");

  cache.emplace(key, fn);
  return fn;
}

inline int8_t ternary_code(float scaled) {
  const float r = std::nearbyintf(scaled);
  if (r > 0.5f) return 1;   // +1
  if (r < -0.5f) return 2;  // -1, two's complement in 2 bits
  return 0;
}

#ifdef XSMM_TERNARY_AVX2

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

// 32 floats -> 32 saturated int8, undoing the lane interleave that the two
// pack stages introduce.
inline __m256i pack32_ps_to_epi8(__m256 a, __m256 b, __m256 c, __m256 d) {
  const __m256i ab =
      _mm256_packs_epi32(_mm256_cvtps_epi32(a), _mm256_cvtps_epi32(b));
  const __m256i cd =
      _mm256_packs_epi32(_mm256_cvtps_epi32(c), _mm256_cvtps_epi32(d));
  const __m256i abcd = _mm256_packs_epi16(ab, cd);
  return _mm256_permutevar8x32_epi32(
      abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

#endif  // XSMM_TERNARY_AVX2

#ifdef XSMM_TERNARY_AVX512

inline __m512 load16_ps(const float* p) { return _mm512_loadu_ps(p); }

inline __m512 load16_ps(const at::Half* p) {
  return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

inline __m512 load16_ps(const at::BFloat16* p) {
  const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

#endif  // XSMM_TERNARY_AVX512

// Symmetric int8 quantization of one group of G activations.
template <typename scalar_t>
inline void quantize_group(
    const scalar_t* __restrict src,
    int8_t* __restrict dst,
    int64_t G,
    float& scale_out) {
#ifdef XSMM_TERNARY_AVX512
  if constexpr (Avx2Loadable<scalar_t>::value) {
    if ((G & 15) == 0) {
      __m512 vmax = _mm512_setzero_ps();
      for (int64_t j = 0; j < G; j += 16) {
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(load16_ps(src + j)));
      }
      const float amax = _mm512_reduce_max_ps(vmax);
      const float scale = amax / 127.0f;
      scale_out = scale;
      const __m512 viscale =
          _mm512_set1_ps(scale != 0.0f ? (1.0f / scale) : 0.0f);

      // vpmovsdb replaces the AVX2 two-stage pack plus lane-fixup permute
      for (int64_t j = 0; j < G; j += 16) {
        const __m512 v = _mm512_mul_ps(load16_ps(src + j), viscale);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + j),
                         _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v)));
      }
      return;
    }
  }
#endif
#ifdef XSMM_TERNARY_AVX2
  if constexpr (Avx2Loadable<scalar_t>::value) {
    if ((G & 31) == 0) {
      const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
      __m256 vmax = _mm256_setzero_ps();
      for (int64_t j = 0; j < G; j += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(load8_ps(src + j), absmask));
      }
      const float amax = hmax_ps(vmax);
      const float scale = amax / 127.0f;
      scale_out = scale;
      const __m256 viscale =
          _mm256_set1_ps(scale != 0.0f ? (1.0f / scale) : 0.0f);

      for (int64_t j = 0; j < G; j += 32) {
        const __m256 v0 = _mm256_mul_ps(load8_ps(src + j + 0), viscale);
        const __m256 v1 = _mm256_mul_ps(load8_ps(src + j + 8), viscale);
        const __m256 v2 = _mm256_mul_ps(load8_ps(src + j + 16), viscale);
        const __m256 v3 = _mm256_mul_ps(load8_ps(src + j + 24), viscale);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst + j),
            pack32_ps_to_epi8(v0, v1, v2, v3));
      }
      return;
    }
  }
#endif
  float amax = 0.0f;
  for (int64_t j = 0; j < G; ++j) {
    amax = std::max(amax, std::fabs(static_cast<float>(src[j])));
  }
  const float scale = amax / 127.0f;
  const float iscale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
  scale_out = scale;
  for (int64_t j = 0; j < G; ++j) {
    const float q = std::nearbyintf(static_cast<float>(src[j]) * iscale);
    dst[j] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
  }
}

// acc[0:32] = int32_acc[0:32] * (act_scale * weight_scale[0:32])
// Used for the first reduction group so the accumulator needs no pre-zeroing.
// Weight scales may be fp16 (half the bandwidth); activation scales stay fp32
// since they are broadcast and would need an upconvert either way.
template <typename wscale_t>
inline void dequant_store(
    const int32_t* __restrict src,
    float* __restrict dst,
    float act_scale,
    const wscale_t* __restrict wsc) {
#ifdef XSMM_TERNARY_AVX512
  const __m512 dx = _mm512_set1_ps(act_scale);
  for (int r = 0; r < kBlockN; r += 16) {
    const __m512 sc = _mm512_mul_ps(dx, load16_ps(wsc + r));
    const __m512 acc = _mm512_cvtepi32_ps(
        _mm512_loadu_si512(reinterpret_cast<const void*>(src + r)));
    _mm512_storeu_ps(dst + r, _mm512_mul_ps(sc, acc));
  }
#elif defined(XSMM_TERNARY_AVX2)
  const __m256 dx = _mm256_set1_ps(act_scale);
  for (int r = 0; r < kBlockN; r += 8) {
    const __m256 sc = _mm256_mul_ps(dx, load8_ps(wsc + r));
    const __m256 acc = _mm256_cvtepi32_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + r)));
    _mm256_storeu_ps(dst + r, _mm256_mul_ps(sc, acc));
  }
#else
  for (int r = 0; r < kBlockN; ++r) {
    dst[r] = static_cast<float>(src[r]) * act_scale * static_cast<float>(wsc[r]);
  }
#endif
}

// acc[0:32] += int32_acc[0:32] * (act_scale * weight_scale[0:32])
template <typename wscale_t>
inline void dequant_accum(
    const int32_t* __restrict src,
    float* __restrict dst,
    float act_scale,
    const wscale_t* __restrict wsc) {
#ifdef XSMM_TERNARY_AVX512
  const __m512 dx = _mm512_set1_ps(act_scale);
  for (int r = 0; r < kBlockN; r += 16) {
    const __m512 sc = _mm512_mul_ps(dx, load16_ps(wsc + r));
    const __m512 acc = _mm512_cvtepi32_ps(
        _mm512_loadu_si512(reinterpret_cast<const void*>(src + r)));
    _mm512_storeu_ps(dst + r,
                     _mm512_fmadd_ps(sc, acc, _mm512_loadu_ps(dst + r)));
  }
#elif defined(XSMM_TERNARY_AVX2)
  const __m256 dx = _mm256_set1_ps(act_scale);
  for (int r = 0; r < kBlockN; r += 8) {
    const __m256 sc = _mm256_mul_ps(dx, load8_ps(wsc + r));
    const __m256 acc = _mm256_cvtepi32_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + r)));
    _mm256_storeu_ps(
        dst + r,
        _mm256_add_ps(_mm256_mul_ps(sc, acc), _mm256_loadu_ps(dst + r)));
  }
#else
  for (int r = 0; r < kBlockN; ++r) {
    dst[r] += static_cast<float>(src[r]) * act_scale * static_cast<float>(wsc[r]);
  }
#endif
}

// Last reduction group: dequantize, add the running accumulator and bias, and
// convert straight to the output dtype, so `acc` is never re-read afterwards.
template <typename scalar_t, typename wscale_t>
inline void dequant_finalize(
    const int32_t* __restrict src,
    const float* __restrict acc,
    bool accumulate,
    float act_scale,
    const wscale_t* __restrict wsc,
    const scalar_t* __restrict bias,
    scalar_t* __restrict dst) {
  alignas(64) float fin[kBlockN];
#ifdef XSMM_TERNARY_AVX512
  const __m512 dx = _mm512_set1_ps(act_scale);
  for (int r = 0; r < kBlockN; r += 16) {
    const __m512 sc = _mm512_mul_ps(dx, load16_ps(wsc + r));
    const __m512 a = _mm512_cvtepi32_ps(
        _mm512_loadu_si512(reinterpret_cast<const void*>(src + r)));
    __m512 v = _mm512_mul_ps(sc, a);
    if (accumulate) {
      v = _mm512_add_ps(v, _mm512_loadu_ps(acc + r));
    }
    _mm512_storeu_ps(fin + r, v);
  }
#elif defined(XSMM_TERNARY_AVX2)
  const __m256 dx = _mm256_set1_ps(act_scale);
  for (int r = 0; r < kBlockN; r += 8) {
    const __m256 sc = _mm256_mul_ps(dx, load8_ps(wsc + r));
    const __m256 a = _mm256_cvtepi32_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + r)));
    __m256 v = _mm256_mul_ps(sc, a);
    if (accumulate) {
      v = _mm256_add_ps(v, _mm256_loadu_ps(acc + r));
    }
    _mm256_storeu_ps(fin + r, v);
  }
#else
  for (int r = 0; r < kBlockN; ++r) {
    const float v =
        static_cast<float>(src[r]) * act_scale * static_cast<float>(wsc[r]);
    fin[r] = accumulate ? acc[r] + v : v;
  }
#endif
  if (bias != nullptr) {
    for (int r = 0; r < kBlockN; ++r) {
      dst[r] = static_cast<scalar_t>(fin[r] + static_cast<float>(bias[r]));
    }
  } else {
    for (int r = 0; r < kBlockN; ++r) {
      dst[r] = static_cast<scalar_t>(fin[r]);
    }
  }
}

// Symmetric per (row, group) int8 quantization of the activations.
template <typename scalar_t>
void quantize_activations(
    const scalar_t* __restrict x,
    int8_t* __restrict xq,
    float* __restrict xs,
    int64_t M,
    int64_t C,
    int64_t G) {
  const int64_t C1 = C / G;

#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t c1 = 0; c1 < C1; ++c1) {
      quantize_group<scalar_t>(
          x + m * C + c1 * G, xq + m * C + c1 * G, G, xs[m * C1 + c1]);
    }
  }
}

struct TernaryGemmPlan {
  libxsmm_gemmfunction k_full;
  libxsmm_gemmfunction k_rem;
  int64_t M, N1, C1, C, G;
  int64_t m_full, m_blocks, m_rem;
  // bytes of packed weight per (32-column block, group)
  int64_t group_bytes;
};

// Counters so the GEMM cost can be isolated from the rest of the model during a
// real end-to-end run. Updated once per op call from the calling thread only.
struct TernaryStats {
  std::atomic<int64_t> calls{0};
  std::atomic<int64_t> rows{0};
  std::atomic<int64_t> weight_bytes{0};
  std::atomic<int64_t> nanos{0};
};

TernaryStats& ternary_stats() {
  static TernaryStats s;
  return s;
}

// Reused across calls so the decode critical path stays allocation-free: at
// ~144 GEMMs per token these temporaries would otherwise be re-allocated
// thousands of times per second.
struct TernaryScratch {
  std::vector<int8_t> xq;
  std::vector<float> xs;
};

TernaryScratch& ternary_scratch() {
  static thread_local TernaryScratch s;
  return s;
}

template <typename scalar_t, typename wscale_t>
void ternary_gemm_tiles(
    const TernaryGemmPlan& plan,
    const uint8_t* __restrict w,
    const wscale_t* __restrict ws,
    const int8_t* __restrict xq,
    const float* __restrict xs,
    const scalar_t* __restrict bias,
    scalar_t* __restrict out) {
  const int64_t N = plan.N1 * kBlockN;
  const int64_t group_bytes = plan.group_bytes;
  const int64_t panel_bytes = plan.C1 * group_bytes;
  const int64_t n_work = plan.N1 * plan.m_blocks;
  const int64_t C1 = plan.C1;

#pragma omp parallel
  {
    // per-thread state: the gemm_param zeroing happens once per thread rather
    // than once per tile, and the accumulators never need pre-zeroing because
    // the first reduction group stores instead of accumulating.
    alignas(64) int32_t acc_i32[kBlockM * kBlockN];
    alignas(64) float acc[kBlockM * kBlockN];

    libxsmm_gemm_param gemm_param;
    std::memset(&gemm_param, 0, sizeof(gemm_param));
    unsigned long long br_count = 1;
    gemm_param.op.tertiary = &br_count;
    gemm_param.c.primary = acc_i32;

// Every tile is the same cost, so on the server parts the shared counter of a
// dynamic schedule costs more than it balances.
#ifdef __AVX512F__
#pragma omp for schedule(static)
#else
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t work = 0; work < n_work; ++work) {
      const int64_t n1 = work / plan.m_blocks;
      const int64_t mb = work % plan.m_blocks;
      const int64_t m0 = mb * plan.m_full;
      const int64_t m_cur =
          (mb + 1 == plan.m_blocks) ? plan.m_rem : plan.m_full;
      const auto kernel = (m_cur == plan.m_full) ? plan.k_full : plan.k_rem;
      const scalar_t* bias_row = (bias != nullptr) ? bias + n1 * kBlockN : nullptr;

      for (int64_t c1 = 0; c1 < C1; ++c1) {
        gemm_param.a.primary =
            const_cast<uint8_t*>(w + n1 * panel_bytes + c1 * group_bytes);
        gemm_param.b.primary =
            const_cast<int8_t*>(xq + m0 * plan.C + c1 * plan.G);
        kernel(&gemm_param);

        const wscale_t* wsc = ws + (n1 * C1 + c1) * kBlockN;
        const bool first = (c1 == 0);
        const bool last = (c1 + 1 == C1);

        for (int64_t m = 0; m < m_cur; ++m) {
          const int32_t* src = acc_i32 + m * kBlockN;
          float* a = acc + m * kBlockN;
          const float xsc = xs[(m0 + m) * C1 + c1];
          if (last) {
            dequant_finalize<scalar_t, wscale_t>(
                src, a, !first, xsc, wsc, bias_row,
                out + (m0 + m) * N + n1 * kBlockN);
          } else if (first) {
            dequant_store<wscale_t>(src, a, xsc, wsc);
          } else {
            dequant_accum<wscale_t>(src, a, xsc, wsc);
          }
        }
      }
    }
  }
}

// Merged gate_up projection with silu(gate)*up folded into the tile epilogue.
// The gate and up halves of a merged projection are 32-column blocks n1 and
// n1 + N1/2, so one work item owns both and the intermediate never reaches
// memory. Saves a parallel region and a read+write of [M, 2I] per layer.
template <typename scalar_t, typename wscale_t>
void ternary_gemm_silu_mul_tiles(
    const TernaryGemmPlan& plan,
    const uint8_t* __restrict w,
    const wscale_t* __restrict ws,
    const int8_t* __restrict xq,
    const float* __restrict xs,
    const scalar_t* __restrict bias,
    scalar_t* __restrict out) {
  const int64_t group_bytes = plan.group_bytes;
  const int64_t panel_bytes = plan.C1 * group_bytes;
  const int64_t C1 = plan.C1;
  const int64_t N1h = plan.N1 / 2;
  const int64_t I = N1h * kBlockN;
  const int64_t n_work = N1h * plan.m_blocks;

#pragma omp parallel
  {
    alignas(64) int32_t acc_i32[kBlockM * kBlockN];
    alignas(64) float res[2][kBlockM * kBlockN];

    libxsmm_gemm_param gemm_param;
    std::memset(&gemm_param, 0, sizeof(gemm_param));
    unsigned long long br_count = 1;
    gemm_param.op.tertiary = &br_count;
    gemm_param.c.primary = acc_i32;

#ifdef __AVX512F__
#pragma omp for schedule(static)
#else
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t work = 0; work < n_work; ++work) {
      const int64_t n1 = work / plan.m_blocks;
      const int64_t mb = work % plan.m_blocks;
      const int64_t m0 = mb * plan.m_full;
      const int64_t m_cur =
          (mb + 1 == plan.m_blocks) ? plan.m_rem : plan.m_full;
      const auto kernel = (m_cur == plan.m_full) ? plan.k_full : plan.k_rem;

      for (int half = 0; half < 2; ++half) {
        const int64_t nb = n1 + half * N1h;
        for (int64_t c1 = 0; c1 < C1; ++c1) {
          gemm_param.a.primary =
              const_cast<uint8_t*>(w + nb * panel_bytes + c1 * group_bytes);
          gemm_param.b.primary =
              const_cast<int8_t*>(xq + m0 * plan.C + c1 * plan.G);
          kernel(&gemm_param);

          const wscale_t* wsc = ws + (nb * C1 + c1) * kBlockN;
          for (int64_t m = 0; m < m_cur; ++m) {
            const int32_t* src = acc_i32 + m * kBlockN;
            float* a = res[half] + m * kBlockN;
            const float xsc = xs[(m0 + m) * C1 + c1];
            if (c1 == 0) {
              dequant_store<wscale_t>(src, a, xsc, wsc);
            } else {
              dequant_accum<wscale_t>(src, a, xsc, wsc);
            }
          }
        }
      }

      const scalar_t* bg = (bias != nullptr) ? bias + n1 * kBlockN : nullptr;
      const scalar_t* bu =
          (bias != nullptr) ? bias + (n1 + N1h) * kBlockN : nullptr;
      for (int64_t m = 0; m < m_cur; ++m) {
        const float* g = res[0] + m * kBlockN;
        const float* u = res[1] + m * kBlockN;
        scalar_t* dst = out + (m0 + m) * I + n1 * kBlockN;
        for (int r = 0; r < kBlockN; ++r) {
          float gv = g[r];
          float uv = u[r];
          if (bias != nullptr) {
            gv += static_cast<float>(bg[r]);
            uv += static_cast<float>(bu[r]);
          }
          dst[r] = static_cast<scalar_t>(gv / (1.0f + std::exp(-gv)) * uv);
        }
      }
    }
  }
}

// One (expert, 32-column block, row block) tile: reduce over all groups and
// dequantize into `out_rows`. Shared by both MoE GEMMs.
template <typename scalar_t, typename wscale_t>
inline void moe_tile(
    libxsmm_gemmfunction kernel,
    libxsmm_gemm_param& gemm_param,
    const uint8_t* __restrict w_panel,
    const wscale_t* __restrict ws_panel,
    const int8_t* __restrict xq_rows,
    const float* __restrict xs_rows,
    int64_t m_cur,
    int64_t C1,
    int64_t G,
    int64_t group_bytes,
    scalar_t* __restrict out_rows,
    int64_t ldo,
    int32_t* __restrict acc_i32,
    float* __restrict acc) {
  for (int64_t c1 = 0; c1 < C1; ++c1) {
    gemm_param.a.primary = const_cast<uint8_t*>(w_panel + c1 * group_bytes);
    gemm_param.b.primary = const_cast<int8_t*>(xq_rows + c1 * G);
    kernel(&gemm_param);

    const wscale_t* wsc = ws_panel + c1 * kBlockN;
    const bool first = (c1 == 0);
    const bool last = (c1 + 1 == C1);
    for (int64_t m = 0; m < m_cur; ++m) {
      const int32_t* src = acc_i32 + m * kBlockN;
      float* a = acc + m * kBlockN;
      const float xsc = xs_rows[m * C1 + c1];
      if (last) {
        dequant_finalize<scalar_t, wscale_t>(
            src, a, !first, xsc, wsc, nullptr, out_rows + m * ldo);
      } else if (first) {
        dequant_store<wscale_t>(src, a, xsc, wsc);
      } else {
        dequant_accum<wscale_t>(src, a, xsc, wsc);
      }
    }
  }
}

// Buffers for the fused MoE op, reused across calls so the decode path stays
// allocation-free.
struct MoeScratch {
  std::vector<int8_t> xq, aq;
  std::vector<float> xs, as;
  std::vector<int8_t> xq_g;
  std::vector<float> xs_g;
  std::vector<char> h, y;
  std::vector<int32_t> perm, inv, offsets, counts;
};

MoeScratch& moe_scratch() {
  static thread_local MoeScratch s;
  return s;
}

// Copy each expert's activation rows together so the GEMM sees them contiguous.
inline void moe_gather(
    MoeScratch& s, int64_t TK, int64_t K, int64_t H, int64_t C1) {
#pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < TK; ++j) {
    const int64_t t = s.perm[j] / K;
    std::memcpy(s.xq_g.data() + j * H, s.xq.data() + t * H, H);
    std::memcpy(
        s.xs_g.data() + j * C1, s.xs.data() + t * C1, C1 * sizeof(float));
  }
}

struct MoeWork {
  int32_t e, blk, mb;
};

struct MoeDims {
  int64_t T, H, K, G, E, N13, I, C1, D1, TK;
  int64_t n13_blocks, h_blocks;
  int64_t w13_panel, w13_expert, w2_panel, w2_expert, group_bytes;
};

// OpenMP directives cannot live inside an AT_DISPATCH macro argument, so the
// whole parallel region sits here and the dispatch just calls it.
//
// The h/y intermediates are float32 rather than the model dtype: rounding the
// gate/up values to 16 bits before SiLU, and again per expert during the
// reduction, costs about 5x accuracy at fp16.
template <typename scalar_t, typename wscale_t>
void moe_run(
    const MoeDims& d,
    MoeScratch& s,
    const std::vector<MoeWork>& a_work,
    const std::vector<MoeWork>& b_work,
    const libxsmm_gemmfunction* __restrict ka,
    const libxsmm_gemmfunction* __restrict kb,
    const uint8_t* __restrict w13_p,
    const wscale_t* __restrict w13s,
    const uint8_t* __restrict w2_p,
    const wscale_t* __restrict w2s,
    const float* __restrict wts_p,
    float* __restrict h_buf,
    float* __restrict y_buf,
    scalar_t* __restrict out_p) {
#pragma omp parallel
  {
    alignas(64) int32_t acc_i32[kBlockM * kBlockN];
    alignas(64) float acc[kBlockM * kBlockN];
    alignas(64) float tmp[kBlockM * kBlockN];
    libxsmm_gemm_param gp;
    std::memset(&gp, 0, sizeof(gp));
    unsigned long long br_count = 1;
    gp.op.tertiary = &br_count;
    gp.c.primary = acc_i32;

#ifdef __AVX512F__
#pragma omp for schedule(static)
#else
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t w = 0; w < static_cast<int64_t>(a_work.size()); ++w) {
      const MoeWork it = a_work[w];
      const int64_t base = s.offsets[it.e];
      const int64_t rows = s.offsets[it.e + 1] - base;
      const int64_t m0 = static_cast<int64_t>(it.mb) * kBlockM;
      const int64_t m_cur = std::min<int64_t>(kBlockM, rows - m0);
      moe_tile<float, wscale_t>(
          ka[m_cur], gp,
          w13_p + it.e * d.w13_expert + it.blk * d.w13_panel,
          w13s + (it.e * d.n13_blocks + it.blk) * d.C1 * kBlockN,
          s.xq_g.data() + (base + m0) * d.H,
          s.xs_g.data() + (base + m0) * d.C1,
          m_cur, d.C1, d.G, d.group_bytes,
          h_buf + (base + m0) * d.N13 + it.blk * kBlockN, d.N13,
          acc_i32, acc);
    }

    // SiLU-gated activation, then requantize for the second GEMM
#pragma omp for schedule(static)
    for (int64_t j = 0; j < d.TK; ++j) {
      const float* hp = h_buf + j * d.N13;
      for (int64_t c = 0; c < d.D1; ++c) {
        for (int64_t i = 0; i < d.G; ++i) {
          const int64_t o = c * d.G + i;
          const float gate = hp[o];
          tmp[i] = (gate / (1.0f + std::exp(-gate))) * hp[d.I + o];
        }
        quantize_group<float>(
            tmp, s.aq.data() + j * d.I + c * d.G, d.G, s.as[j * d.D1 + c]);
      }
    }

#ifdef __AVX512F__
#pragma omp for schedule(static)
#else
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t w = 0; w < static_cast<int64_t>(b_work.size()); ++w) {
      const MoeWork it = b_work[w];
      const int64_t base = s.offsets[it.e];
      const int64_t rows = s.offsets[it.e + 1] - base;
      const int64_t m0 = static_cast<int64_t>(it.mb) * kBlockM;
      const int64_t m_cur = std::min<int64_t>(kBlockM, rows - m0);
      moe_tile<float, wscale_t>(
          kb[m_cur], gp,
          w2_p + it.e * d.w2_expert + it.blk * d.w2_panel,
          w2s + (it.e * d.h_blocks + it.blk) * d.D1 * kBlockN,
          s.aq.data() + (base + m0) * d.I,
          s.as.data() + (base + m0) * d.D1,
          m_cur, d.D1, d.G, d.group_bytes,
          y_buf + (base + m0) * d.H + it.blk * kBlockN, d.H,
          acc_i32, acc);
    }

    // each token is owned by one thread, so the reduction needs no atomics
#pragma omp for schedule(static)
    for (int64_t t = 0; t < d.T; ++t) {
      scalar_t* o = out_p + t * d.H;
      for (int64_t c = 0; c < d.H; ++c) {
        float sum = 0.0f;
        for (int64_t k = 0; k < d.K; ++k) {
          sum += y_buf[s.inv[t * d.K + k] * d.H + c] * wts_p[t * d.K + k];
        }
        o[c] = static_cast<scalar_t>(sum);
      }
    }
  }
}

}  // namespace

// Fused ternary MoE: y[t] = sum_k topk_weight[t,k] *
//     down_e( silu(gate_e(x[t])) * up_e(x[t]) ),  e = topk_ids[t,k]
//
// Doing this in one op keeps the decode path off the Python interpreter: a
// per-expert loop issues two GEMM calls per expert per layer, and at 48 layers
// x 8 experts that overhead dominates the actual weight streaming.
torch::Tensor xsmm_ternary_moe(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    int64_t group_size) {
  TORCH_CHECK(input.dim() == 2, "input must be [T, H]");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.dim() == 2, "topk must be 2-D");
  TORCH_CHECK(w13.dim() == 6 && w2.dim() == 6, "packed MoE weights must be 6-D");
  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt || topk_ids.scalar_type() == torch::kLong,
      "topk_ids must be int32 or int64");

  const auto t_start = std::chrono::steady_clock::now();

  const auto x = input.contiguous();
  const auto ids = topk_ids.to(torch::kInt).contiguous();
  const auto wts = topk_weights.to(torch::kFloat).contiguous();

  const int64_t T = x.size(0);
  const int64_t H = x.size(1);
  const int64_t K = ids.size(1);
  const int64_t G = group_size;
  const int64_t E = w13.size(0);
  const int64_t N13 = w13.size(1) * kBlockN;  // 2 * intermediate
  const int64_t I = N13 / 2;
  const int64_t C1 = H / G;
  const int64_t D1 = I / G;
  const int64_t TK = T * K;

  TORCH_CHECK(C1 * G == H, "hidden size must be a multiple of group_size");
  TORCH_CHECK(D1 * G == I, "intermediate size must be a multiple of group_size");
  TORCH_CHECK(w2.size(1) * kBlockN == H, "w2 output does not match hidden size");

  auto out = torch::zeros({T, H}, x.options());

  const int64_t w13_group_bytes = (G / kVnni) * kLanes * kVnni;
  const int64_t w13_panel = C1 * w13_group_bytes;
  const int64_t w13_expert = (N13 / kBlockN) * w13_panel;
  const int64_t w2_group_bytes = w13_group_bytes;
  const int64_t w2_panel = D1 * w2_group_bytes;
  const int64_t w2_expert = (H / kBlockN) * w2_panel;

  auto& s = moe_scratch();
  s.xq.resize(T * H);
  s.xs.resize(T * C1);
  s.xq_g.resize(TK * H);
  s.xs_g.resize(TK * C1);
  s.aq.resize(TK * I);
  s.as.resize(TK * D1);
  s.perm.resize(TK);
  s.inv.resize(TK);
  s.counts.assign(E + 1, 0);
  s.offsets.assign(E + 1, 0);

  const int32_t* ids_p = ids.data_ptr<int32_t>();
  const float* wts_p = wts.data_ptr<float>();

  // counting sort of the (token, slot) pairs by expert
  for (int64_t p = 0; p < TK; ++p) {
    s.counts[ids_p[p] + 1]++;
  }
  for (int64_t e = 0; e < E; ++e) {
    s.counts[e + 1] += s.counts[e];
  }
  std::copy(s.counts.begin(), s.counts.end(), s.offsets.begin());
  {
    std::vector<int32_t> pos(s.counts.begin(), s.counts.end());
    for (int64_t p = 0; p < TK; ++p) {
      const int32_t e = ids_p[p];
      const int32_t j = pos[e]++;
      s.perm[j] = static_cast<int32_t>(p);
      s.inv[p] = j;
    }
  }

  const int64_t n13_blocks = N13 / kBlockN;
  const int64_t h_blocks = H / kBlockN;

  MoeDims d{T,  H,  K,  G,  E,  N13, I, C1, D1, TK,
            n13_blocks, h_blocks,
            w13_panel, w13_expert, w2_panel, w2_expert, w13_group_bytes};

  // work items: (expert, 32-column block, row block) for each of the two GEMMs
  std::vector<MoeWork> a_work, b_work;
  int64_t max_rows = 0;
  for (int64_t e = 0; e < E; ++e) {
    const int64_t rows = s.offsets[e + 1] - s.offsets[e];
    if (rows == 0) continue;
    max_rows = std::max(max_rows, rows);
    const int64_t mblocks = (rows + kBlockM - 1) / kBlockM;
    for (int64_t b = 0; b < n13_blocks; ++b)
      for (int64_t m = 0; m < mblocks; ++m)
        a_work.push_back({(int32_t)e, (int32_t)b, (int32_t)m});
    for (int64_t b = 0; b < h_blocks; ++b)
      for (int64_t m = 0; m < mblocks; ++m)
        b_work.push_back({(int32_t)e, (int32_t)b, (int32_t)m});
  }

  // Resolve the JIT kernels up front: the lookup takes a process-global mutex,
  // so doing it per work item would serialize every thread.
  std::vector<libxsmm_gemmfunction> ka(kBlockM + 1, nullptr);
  std::vector<libxsmm_gemmfunction> kb(kBlockM + 1, nullptr);
  for (int64_t e = 0; e < E; ++e) {
    const int64_t rows = s.offsets[e + 1] - s.offsets[e];
    if (rows == 0) continue;
    for (int64_t m0 = 0; m0 < rows; m0 += kBlockM) {
      const int64_t m_cur = std::min<int64_t>(kBlockM, rows - m0);
      if (ka[m_cur] == nullptr) {
        ka[m_cur] = get_ternary_kernel(
            {(int32_t)kBlockN, (int32_t)m_cur, (int32_t)G, (int32_t)kBlockN,
             (int32_t)H, (int32_t)kBlockN, 2});
        kb[m_cur] = get_ternary_kernel(
            {(int32_t)kBlockN, (int32_t)m_cur, (int32_t)G, (int32_t)kBlockN,
             (int32_t)I, (int32_t)kBlockN, 2});
      }
    }
  }
  (void)max_rows;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "xsmm_ternary_moe", [&] {
        s.h.resize(TK * N13 * sizeof(float));
        s.y.resize(TK * H * sizeof(float));
        auto* h_buf = reinterpret_cast<float*>(s.h.data());
        auto* y_buf = reinterpret_cast<float*>(s.y.data());
        const scalar_t* x_p = x.data_ptr<scalar_t>();
        scalar_t* out_p = out.data_ptr<scalar_t>();

        quantize_activations<scalar_t>(x_p, s.xq.data(), s.xs.data(), T, H, G);
        moe_gather(s, TK, K, H, C1);

        const uint8_t* w13_p = w13.data_ptr<uint8_t>();
        const uint8_t* w2_p = w2.data_ptr<uint8_t>();

        if (w13_scale.scalar_type() == torch::kHalf) {
          moe_run<scalar_t, at::Half>(
              d, s, a_work, b_work, ka.data(), kb.data(), w13_p,
              w13_scale.data_ptr<at::Half>(), w2_p,
              w2_scale.data_ptr<at::Half>(), wts_p, h_buf, y_buf, out_p);
        } else {
          moe_run<scalar_t, float>(
              d, s, a_work, b_work, ka.data(), kb.data(), w13_p,
              w13_scale.data_ptr<float>(), w2_p, w2_scale.data_ptr<float>(),
              wts_p, h_buf, y_buf, out_p);
        }
      });

  auto& st = ternary_stats();
  st.calls.fetch_add(1, std::memory_order_relaxed);
  st.rows.fetch_add(T, std::memory_order_relaxed);
  st.nanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_start)
          .count(),
      std::memory_order_relaxed);
  return out;
}

// Quantize [N, C] float weights to ternary and pack them into the layout the
// LIBXSMM int2 microkernel expects.
//
// Returns:
//   packed [N/32, C/G, G/4, 8, 4] uint8 -- byte b packs the 4 output rows
//                                          n = n1*32 + k3*8 + k2 at bits 2*k3
//   scales [N/32, C/G, 32]  float or half -- absmax per (output row, group)
std::tuple<torch::Tensor, torch::Tensor> xsmm_ternary_pack_weight(
    const torch::Tensor& weight,
    int64_t group_size,
    bool half_scales) {
  TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, C]");
  const int64_t N = weight.size(0);
  const int64_t C = weight.size(1);
  const int64_t G = group_size;

  TORCH_CHECK(N % kBlockN == 0, "N (", N, ") must be a multiple of ", kBlockN);
  TORCH_CHECK(G % kVnni == 0, "group_size (", G, ") must be a multiple of ", kVnni);
  TORCH_CHECK(C % G == 0, "C (", C, ") must be a multiple of group_size (", G, ")");

  const auto w = weight.to(torch::kFloat).contiguous();
  const int64_t N1 = N / kBlockN;
  const int64_t C1 = C / G;
  const int64_t C2 = G / kVnni;

  auto packed = torch::empty(
      {N1, C1, C2, kLanes, kVnni}, weight.options().dtype(torch::kUInt8));
  auto scales =
      torch::empty({N1, C1, kBlockN}, weight.options().dtype(torch::kFloat));

  const float* w_ptr = w.data_ptr<float>();
  uint8_t* p_ptr = packed.data_ptr<uint8_t>();
  float* s_ptr = scales.data_ptr<float>();

  const int64_t tiles = N1 * C1;

#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t t = 0; t < tiles; ++t) {
    const int64_t n1 = t / C1;
    const int64_t c1 = t % C1;

    float* tile_scales = s_ptr + t * kBlockN;
    float inv_scale[kBlockN];

    for (int64_t r = 0; r < kBlockN; ++r) {
      const float* row = w_ptr + (n1 * kBlockN + r) * C + c1 * G;
      float amax = 0.0f;
      for (int64_t j = 0; j < G; ++j) {
        amax = std::max(amax, std::fabs(row[j]));
      }
      const float scale = std::max(amax, 1e-8f);
      tile_scales[r] = scale;
      inv_scale[r] = 1.0f / scale;
    }

    uint8_t* tile = p_ptr + t * C2 * kLanes * kVnni;
    for (int64_t c2 = 0; c2 < C2; ++c2) {
      for (int64_t k2 = 0; k2 < kLanes; ++k2) {
        for (int64_t c3 = 0; c3 < kVnni; ++c3) {
          const int64_t c = c1 * G + c2 * kVnni + c3;
          uint8_t byte = 0;
          for (int64_t k3 = 0; k3 < kPack; ++k3) {
            const int64_t r = k3 * kLanes + k2;
            const float v = w_ptr[(n1 * kBlockN + r) * C + c] * inv_scale[r];
            byte |= static_cast<uint8_t>(ternary_code(v)) << (2 * k3);
          }
          tile[(c2 * kLanes + k2) * kVnni + c3] = byte;
        }
      }
    }
  }

  return {packed, half_scales ? scales.to(torch::kHalf) : scales};
}

// Quantize [N, C] float weights to symmetric per-(row, group) int8 and pack
// them as plain VNNI4 -- the same tiling as the int2 path but one byte per
// weight, so 32 rows sit where the int2 layout puts 8 lanes x 4 bit-planes.
//
// Returns:
//   packed [N/32, C/G, G/4, 32, 4] int8
//   scales [N/32, C/G, 32] float or half -- absmax/127 per (output row, group)
std::tuple<torch::Tensor, torch::Tensor> xsmm_int8_pack_weight(
    const torch::Tensor& weight,
    int64_t group_size,
    bool half_scales) {
  TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, C]");
  const int64_t N = weight.size(0);
  const int64_t C = weight.size(1);
  const int64_t G = group_size;

  TORCH_CHECK(N % kBlockN == 0, "N (", N, ") must be a multiple of ", kBlockN);
  TORCH_CHECK(G % kVnni == 0, "group_size (", G, ") must be a multiple of ", kVnni);
  TORCH_CHECK(C % G == 0, "C (", C, ") must be a multiple of group_size (", G, ")");

  const auto w = weight.to(torch::kFloat).contiguous();
  const int64_t N1 = N / kBlockN;
  const int64_t C1 = C / G;
  const int64_t C2 = G / kVnni;

  auto packed = torch::empty(
      {N1, C1, C2, kBlockN, kVnni}, weight.options().dtype(torch::kChar));
  auto scales =
      torch::empty({N1, C1, kBlockN}, weight.options().dtype(torch::kFloat));

  const float* w_ptr = w.data_ptr<float>();
  int8_t* p_ptr = packed.data_ptr<int8_t>();
  float* s_ptr = scales.data_ptr<float>();

  const int64_t tiles = N1 * C1;

#ifdef __AVX512F__
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t t = 0; t < tiles; ++t) {
    const int64_t n1 = t / C1;
    const int64_t c1 = t % C1;

    float* tile_scales = s_ptr + t * kBlockN;
    float inv_scale[kBlockN];

    for (int64_t r = 0; r < kBlockN; ++r) {
      const float* row = w_ptr + (n1 * kBlockN + r) * C + c1 * G;
      float amax = 0.0f;
      for (int64_t j = 0; j < G; ++j) {
        amax = std::max(amax, std::fabs(row[j]));
      }
      const float scale = std::max(amax / 127.0f, 1e-12f);
      tile_scales[r] = scale;
      inv_scale[r] = 1.0f / scale;
    }

    int8_t* tile = p_ptr + t * C2 * kBlockN * kVnni;
    for (int64_t c2 = 0; c2 < C2; ++c2) {
      for (int64_t r = 0; r < kBlockN; ++r) {
        for (int64_t c3 = 0; c3 < kVnni; ++c3) {
          const int64_t c = c1 * G + c2 * kVnni + c3;
          const float v = w_ptr[(n1 * kBlockN + r) * C + c] * inv_scale[r];
          const float q = std::nearbyintf(v);
          tile[(c2 * kBlockN + r) * kVnni + c3] = static_cast<int8_t>(
              std::min(127.0f, std::max(-127.0f, q)));
        }
      }
    }
  }

  return {packed, half_scales ? scales.to(torch::kHalf) : scales};
}

// y = x @ dequant(packed)^T (+ bias), written into `out`.
// wbits selects the weight encoding: 2 = int2 interleaved, 8 = VNNI4 int8.
static void quant_gemm_into(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor& out,
    int wbits,
    bool fuse_silu_mul = false) {
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dims");
  TORCH_CHECK(packed.dim() == 5, "packed weight must be 5-D");
  TORCH_CHECK(
      packed.scalar_type() ==
          (wbits == 2 ? torch::kUInt8 : torch::kChar),
      "packed weight dtype does not match the ", wbits, "-bit layout");
  TORCH_CHECK(
      scales.dim() == 3 &&
          (scales.scalar_type() == torch::kFloat ||
           scales.scalar_type() == torch::kHalf),
      "weight scales must be 3-D float32 or float16");

  const auto t_start = std::chrono::steady_clock::now();

  const auto x = input.contiguous();
  const int64_t C = x.size(-1);
  const int64_t M = x.numel() / C;
  const int64_t G = group_size;
  const int64_t N1 = packed.size(0);
  const int64_t C1 = packed.size(1);
  const int64_t N = N1 * kBlockN;

  TORCH_CHECK(C1 * G == C, "activation width (", C, ") does not match packed weight");
  TORCH_CHECK(
      scales.size(0) == N1 && scales.size(1) == C1 && scales.size(2) == kBlockN,
      "scales shape mismatch");
  TORCH_CHECK(!fuse_silu_mul || N1 % 2 == 0,
              "fused silu_mul needs an even number of 32-column blocks");
  const int64_t N_out = fuse_silu_mul ? N / 2 : N;
  TORCH_CHECK(
      out.numel() == M * N_out && out.is_contiguous() &&
          out.scalar_type() == x.scalar_type(),
      "out must be contiguous with ", M * N_out, " elements of the input dtype");

  // Activations: symmetric per (row, group) int8, matching the weight grouping.
  auto& scratch = ternary_scratch();
  if (static_cast<int64_t>(scratch.xq.size()) < M * C) {
    scratch.xq.resize(M * C);
  }
  if (static_cast<int64_t>(scratch.xs.size()) < M * C1) {
    scratch.xs.resize(M * C1);
  }
  int8_t* xq_ptr = scratch.xq.data();
  float* xs_ptr = scratch.xs.data();

  const uint8_t* w_ptr =
      reinterpret_cast<const uint8_t*>(packed.const_data_ptr());
  const bool half_ws = scales.scalar_type() == torch::kHalf;

  // JIT lookup happens once per call, outside the parallel region.
  const int32_t m_full = static_cast<int32_t>(std::min<int64_t>(kBlockM, M));
  const int64_t m_blocks = (M + m_full - 1) / m_full;
  const int32_t m_rem = static_cast<int32_t>(M - (m_blocks - 1) * m_full);

  const GemmKey key_full{
      static_cast<int32_t>(kBlockN),
      m_full,
      static_cast<int32_t>(G),
      static_cast<int32_t>(kBlockN),
      static_cast<int32_t>(C),
      static_cast<int32_t>(kBlockN),
      wbits};

  TernaryGemmPlan plan;
  plan.k_full = get_ternary_kernel(key_full);
  plan.k_rem = plan.k_full;
  if (m_rem != m_full) {
    GemmKey key_rem = key_full;
    key_rem.n = m_rem;
    plan.k_rem = get_ternary_kernel(key_rem);
  }
  plan.M = M;
  plan.N1 = N1;
  plan.C1 = C1;
  plan.C = C;
  plan.G = G;
  plan.m_full = m_full;
  plan.m_blocks = m_blocks;
  plan.m_rem = m_rem;
  plan.group_bytes =
      (G / kVnni) * (wbits == 2 ? kLanes : kBlockN) * kVnni;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "xsmm_ternary_gemm", [&] {
        at::Tensor bias_c;
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
          bias_c = bias->contiguous();
          bias_ptr = bias_c.data_ptr<scalar_t>();
        }
        quantize_activations<scalar_t>(
            x.data_ptr<scalar_t>(), xq_ptr, xs_ptr, M, C, G);
        if (fuse_silu_mul) {
          if (half_ws) {
            ternary_gemm_silu_mul_tiles<scalar_t, at::Half>(
                plan, w_ptr, scales.data_ptr<at::Half>(), xq_ptr, xs_ptr,
                bias_ptr, out.data_ptr<scalar_t>());
          } else {
            ternary_gemm_silu_mul_tiles<scalar_t, float>(
                plan, w_ptr, scales.data_ptr<float>(), xq_ptr, xs_ptr,
                bias_ptr, out.data_ptr<scalar_t>());
          }
        } else if (half_ws) {
          ternary_gemm_tiles<scalar_t, at::Half>(
              plan, w_ptr, scales.data_ptr<at::Half>(), xq_ptr, xs_ptr,
              bias_ptr, out.data_ptr<scalar_t>());
        } else {
          ternary_gemm_tiles<scalar_t, float>(
              plan, w_ptr, scales.data_ptr<float>(), xq_ptr, xs_ptr, bias_ptr,
              out.data_ptr<scalar_t>());
        }
      });

  auto& st = ternary_stats();
  st.calls.fetch_add(1, std::memory_order_relaxed);
  st.rows.fetch_add(M, std::memory_order_relaxed);
  st.weight_bytes.fetch_add(
      packed.numel() + scales.numel() * scales.element_size(),
      std::memory_order_relaxed);
  st.nanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t_start)
          .count(),
      std::memory_order_relaxed);
}

torch::Tensor xsmm_ternary_gemm(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size) {
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = packed.size(0) * kBlockN;
  auto out = torch::empty(out_sizes, input.options());
  quant_gemm_into(input, packed, scales, bias, group_size, out, /*wbits=*/2);
  return out;
}

void xsmm_ternary_gemm_out(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor& out) {
  quant_gemm_into(input, packed, scales, bias, group_size, out, /*wbits=*/2);
}

torch::Tensor xsmm_ternary_gemm_silu_mul(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size) {
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = packed.size(0) * kBlockN / 2;
  auto out = torch::empty(out_sizes, input.options());
  quant_gemm_into(input, packed, scales, bias, group_size, out, /*wbits=*/2,
                  /*fuse_silu_mul=*/true);
  return out;
}

torch::Tensor xsmm_int8_gemm(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size) {
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = packed.size(0) * kBlockN;
  auto out = torch::empty(out_sizes, input.options());
  quant_gemm_into(input, packed, scales, bias, group_size, out, /*wbits=*/8);
  return out;
}

void xsmm_int8_gemm_out(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor& out) {
  quant_gemm_into(input, packed, scales, bias, group_size, out, /*wbits=*/8);
}

// Lets the bitmap kernel in xsmm_ternary_bitmap.cpp share these counters.
void xsmm_ternary_record(int64_t rows, int64_t bytes, int64_t nanos) {
  auto& st = ternary_stats();
  st.calls.fetch_add(1, std::memory_order_relaxed);
  st.rows.fetch_add(rows, std::memory_order_relaxed);
  st.weight_bytes.fetch_add(bytes, std::memory_order_relaxed);
  st.nanos.fetch_add(nanos, std::memory_order_relaxed);
}

// (calls, rows, weight_bytes, seconds) accumulated since the last reset.
std::tuple<int64_t, int64_t, int64_t, double> xsmm_ternary_stats() {
  auto& st = ternary_stats();
  return {
      st.calls.load(std::memory_order_relaxed),
      st.rows.load(std::memory_order_relaxed),
      st.weight_bytes.load(std::memory_order_relaxed),
      st.nanos.load(std::memory_order_relaxed) * 1e-9};
}

void xsmm_ternary_reset_stats() {
  auto& st = ternary_stats();
  st.calls.store(0, std::memory_order_relaxed);
  st.rows.store(0, std::memory_order_relaxed);
  st.weight_bytes.store(0, std::memory_order_relaxed);
  st.nanos.store(0, std::memory_order_relaxed);
}

std::string xsmm_target_arch() {
  return std::string(libxsmm_get_target_arch());
}

// Ask for 2 MiB pages over a weight buffer. A multi-GiB model mapped with 4 KiB
// pages exceeds the L2 TLB by orders of magnitude, so streaming the weights
// costs a page walk per access; the node has THP on `madvise` defrag, which
// means large mappings are not promoted unless asked. Call before filling, so
// the pages fault in already huge.
void xsmm_hugepage_advise(torch::Tensor& t) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  constexpr uintptr_t kHuge = 2u << 20;
  const auto start = reinterpret_cast<uintptr_t>(t.data_ptr());
  const size_t bytes = t.numel() * t.element_size();
  const uintptr_t aligned = (start + kHuge - 1) & ~(kHuge - 1);
  if (bytes > aligned - start + kHuge) {
    madvise(
        reinterpret_cast<void*>(aligned),
        (bytes - (aligned - start)) & ~(kHuge - 1),
        MADV_HUGEPAGE);
  }
#endif
}
