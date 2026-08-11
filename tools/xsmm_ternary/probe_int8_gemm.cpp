// Probe: does LIBXSMM dispatch an int8 x int8 -> int32 VNNI4 BRGEMM on this
// arch, and is the weight layout what we expect?
//
//   g++ -O3 -std=c++17 -mavx2 -DLIBXSMM_DEFAULT_CONFIG -I<libxsmm>/include \
//       probe_int8_gemm.cpp <libxsmm.a> -lpthread -ldl -lm -o probe_int8
//
// Layout under test (per 32-column block, per group of G reduction elements):
//   packed[(c2 * 32 + r) * 4 + c3] = int8 weight[row r][channel c2*4 + c3]
// which is plain VNNI4, i.e. what the int2 path uses with 32 rows instead of
// 8 lanes x 4 bit-planes.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <libxsmm.h>

// LIBXSMM pulls in BLAS for fallback paths that are never called here.
extern "C" {
void dgemm_() {}
void sgemm_() {}
void dgemv_() {}
void sgemv_() {}
}

namespace {

constexpr int kBlockN = 32;  // output columns per tile
constexpr int kVnni = 4;

int check(int nrows, int G) {
  const int C = G;  // single group for the probe
  std::vector<int8_t> w(kBlockN * C);
  std::vector<int8_t> b(nrows * C);
  std::vector<int8_t> a(C / kVnni * kBlockN * kVnni);
  std::vector<int32_t> c(nrows * kBlockN, 0);

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> d(-127, 127);
  for (auto& v : w) v = static_cast<int8_t>(d(rng));
  for (auto& v : b) v = static_cast<int8_t>(d(rng));

  // pack weights into VNNI4
  for (int c2 = 0; c2 < C / kVnni; ++c2) {
    for (int r = 0; r < kBlockN; ++r) {
      for (int c3 = 0; c3 < kVnni; ++c3) {
        a[(c2 * kBlockN + r) * kVnni + c3] = w[r * C + c2 * kVnni + c3];
      }
    }
  }

  libxsmm_gemm_shape shape;
  std::memset(&shape, 0, sizeof(shape));
  shape.m = kBlockN;
  shape.n = nrows;
  shape.k = G;
  shape.lda = kBlockN;
  shape.ldb = C;
  shape.ldc = kBlockN;
  shape.a_in_type = LIBXSMM_DATATYPE_I8;
  shape.b_in_type = LIBXSMM_DATATYPE_I8;
  shape.out_type = LIBXSMM_DATATYPE_I32;
  shape.comp_type = LIBXSMM_DATATYPE_I32;

  const libxsmm_bitfield flags = LIBXSMM_GEMM_FLAGS('N', 'N') |
                                 LIBXSMM_GEMM_FLAG_VNNI_A |
                                 LIBXSMM_GEMM_FLAG_BETA_0;

  libxsmm_gemm_batch_reduce_config br;
  std::memset(&br, 0, sizeof(br));
  br.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  br.br_stride_a_hint = static_cast<libxsmm_blasint>(G) * kBlockN;
  br.br_stride_b_hint = static_cast<libxsmm_blasint>(G);
  br.br_unroll_hint = 1;

  libxsmm_gemmfunction k =
      libxsmm_dispatch_brgemm(shape, flags, LIBXSMM_PREFETCH_NONE, br);
  if (k == nullptr) {
    std::printf("  nrows=%-3d G=%-4d  DISPATCH FAILED\n", nrows, G);
    return 1;
  }

  libxsmm_gemm_param p;
  std::memset(&p, 0, sizeof(p));
  unsigned long long br_count = 1;
  p.op.tertiary = &br_count;
  p.a.primary = a.data();
  p.b.primary = b.data();
  p.c.primary = c.data();
  k(&p);

  int bad = 0;
  int32_t first_ref = 0, first_got = 0;
  for (int m = 0; m < nrows; ++m) {
    for (int r = 0; r < kBlockN; ++r) {
      int32_t ref = 0;
      for (int ch = 0; ch < C; ++ch) {
        ref += static_cast<int32_t>(w[r * C + ch]) *
               static_cast<int32_t>(b[m * C + ch]);
      }
      const int32_t got = c[m * kBlockN + r];
      if (ref != got) {
        if (bad == 0) {
          first_ref = ref;
          first_got = got;
        }
        ++bad;
      }
    }
  }
  std::printf("  nrows=%-3d G=%-4d  %s", nrows, G,
              bad == 0 ? "PASS\n" : "");
  if (bad) {
    std::printf("MISMATCH (%d/%d, first ref=%d got=%d)\n", bad,
                nrows * kBlockN, first_ref, first_got);
  }
  return bad ? 1 : 0;
}

}  // namespace

int main() {
  libxsmm_init();
  std::printf("libxsmm target arch: %s\n", libxsmm_get_target_arch());
  int fails = 0;
  for (int G : {64, 128, 256}) {
    for (int n : {1, 2, 8, 32}) {
      fails += check(n, G);
    }
  }
  std::printf("%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
  return fails ? 1 : 0;
}
