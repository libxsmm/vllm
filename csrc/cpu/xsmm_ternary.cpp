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

  bool operator==(const GemmKey& o) const {
    return m == o.m && n == o.n && k == o.k && lda == o.lda && ldb == o.ldb &&
           ldc == o.ldc;
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

  const libxsmm_bitfield flags = LIBXSMM_GEMM_FLAGS('N', 'N') |
      LIBXSMM_GEMM_FLAG_VNNI_A |
      LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT2_VNNI4_INTLV |
      LIBXSMM_GEMM_FLAG_BETA_0;

  libxsmm_gemm_batch_reduce_config brconfig;
  std::memset(&brconfig, 0, sizeof(brconfig));
  brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  // 4 ternary values per byte
  brconfig.br_stride_a_hint = key.k * key.lda / kPack;
  brconfig.br_stride_b_hint = key.k;
  brconfig.br_unroll_hint = 1;

  libxsmm_gemmfunction fn =
      libxsmm_dispatch_brgemm(shape, flags, /*prefetch=*/0, brconfig);
  TORCH_CHECK(
      fn != nullptr,
      "LIBXSMM failed to JIT an int2xint8 kernel (m=",
      key.m,
      " n=",
      key.n,
      " k=",
      key.k,
      "). The AVX2/SRF int2 microkernel requires AVX-VNNI-INT8 and "
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

// Symmetric int8 quantization of one group of G activations.
template <typename scalar_t>
inline void quantize_group(
    const scalar_t* __restrict src,
    int8_t* __restrict dst,
    int64_t G,
    float& scale_out) {
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
#ifdef XSMM_TERNARY_AVX2
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
#ifdef XSMM_TERNARY_AVX2
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
#ifdef XSMM_TERNARY_AVX2
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

#pragma omp parallel for schedule(dynamic, 1)
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
  const int64_t group_bytes = (plan.G / kVnni) * kLanes * kVnni;
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

#pragma omp for schedule(dynamic, 1)
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

}  // namespace

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

#pragma omp parallel for schedule(dynamic, 1)
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

// y = x @ dequant(packed)^T (+ bias), written into `out`.
void xsmm_ternary_gemm_into(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor& out) {
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dims");
  TORCH_CHECK(packed.dim() == 5 && packed.scalar_type() == torch::kUInt8);
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
  TORCH_CHECK(
      out.numel() == M * N && out.is_contiguous() &&
          out.scalar_type() == x.scalar_type(),
      "out must be contiguous with ", M * N, " elements of the input dtype");

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

  const uint8_t* w_ptr = packed.data_ptr<uint8_t>();
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
      static_cast<int32_t>(kBlockN)};

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
        if (half_ws) {
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
  xsmm_ternary_gemm_into(input, packed, scales, bias, group_size, out);
  return out;
}

void xsmm_ternary_gemm_out(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor& out) {
  xsmm_ternary_gemm_into(input, packed, scales, bias, group_size, out);
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
