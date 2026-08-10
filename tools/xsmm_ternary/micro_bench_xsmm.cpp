// Same timing loop as micro_bench.cpp, but driving LIBXSMM's JIT int2 kernel,
// so the two microkernels can be compared in isolation from the vLLM tile code.
//
//   g++ -O3 -std=c++17 -mavx2 -DLIBXSMM_DEFAULT_CONFIG -I<libxsmm>/include \
//       micro_bench_xsmm.cpp <libxsmm.a> -lpthread -ldl -lm -o mb_xsmm

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <libxsmm.h>

// LIBXSMM references BLAS for its fallback paths; never called here.
extern "C" {
void dgemm_() {}
void sgemm_() {}
void dgemv_() {}
void sgemv_() {}
}

namespace {

constexpr int64_t kK = 128;
constexpr int64_t kC1 = 32;
constexpr int64_t kTiles = 64;
constexpr int64_t kN = 32;  // output columns

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

double run(int nrows, int64_t iters) {
  const int64_t C = kC1 * kK;
  const int64_t group_bytes = (kK / 4) * kN;  // 1024
  const int64_t panel_bytes = kC1 * group_bytes;

  std::vector<uint8_t> w(kTiles * panel_bytes);
  std::vector<int8_t> b(nrows * C);
  std::vector<int32_t> acc(nrows * kN * kTiles, 0);
  for (size_t i = 0; i < w.size(); ++i) w[i] = static_cast<uint8_t>(i * 37u);
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<int8_t>((i % 251) - 125);

  libxsmm_init();

  libxsmm_gemm_shape shape;
  shape.m = static_cast<libxsmm_blasint>(kN);
  shape.n = static_cast<libxsmm_blasint>(nrows);
  shape.k = static_cast<libxsmm_blasint>(kK);
  shape.lda = static_cast<libxsmm_blasint>(kN);
  shape.ldb = static_cast<libxsmm_blasint>(C);
  shape.ldc = static_cast<libxsmm_blasint>(kN);
  shape.a_in_type = LIBXSMM_DATATYPE_I8;
  shape.b_in_type = LIBXSMM_DATATYPE_I8;
  shape.out_type = LIBXSMM_DATATYPE_I32;
  shape.comp_type = LIBXSMM_DATATYPE_I32;

  const libxsmm_bitfield flags =
      LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT2_VNNI4_INTLV |
      LIBXSMM_GEMM_FLAG_VNNI_A | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_gemm_batch_reduce_config br;
  br.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  br.br_stride_a_hint = static_cast<libxsmm_blasint>(group_bytes);
  br.br_stride_b_hint = static_cast<libxsmm_blasint>(kK);
  br.br_unroll_hint = 1;

  libxsmm_gemmfunction kernel =
      libxsmm_dispatch_brgemm(shape, flags, LIBXSMM_PREFETCH_NONE, br);
  if (kernel == nullptr) {
    std::printf("  libxsmm dispatch FAILED (nrows=%d)\n", nrows);
    return 0.0;
  }

  libxsmm_gemm_param p;
  std::memset(&p, 0, sizeof(p));
  unsigned long long br_count = 1;
  p.op.tertiary = &br_count;

  for (int64_t t = 0; t < kTiles; ++t) {
    for (int64_t c1 = 0; c1 < kC1; ++c1) {
      p.a.primary = w.data() + t * panel_bytes + c1 * group_bytes;
      p.b.primary = b.data() + c1 * kK;
      p.c.primary = acc.data() + t * nrows * kN;
      kernel(&p);
    }
  }

  const double t0 = now_s();
  for (int64_t it = 0; it < iters; ++it) {
    for (int64_t t = 0; t < kTiles; ++t) {
      for (int64_t c1 = 0; c1 < kC1; ++c1) {
        p.a.primary = w.data() + t * panel_bytes + c1 * group_bytes;
        p.b.primary = b.data() + c1 * kK;
        p.c.primary = acc.data() + t * nrows * kN;
        kernel(&p);
      }
    }
  }
  const double dt = now_s() - t0;

  int64_t sink = 0;
  for (size_t i = 0; i < acc.size(); ++i) sink += acc[i];
  if (sink == 0x7fffffffffffffffLL) std::fprintf(stderr, "sink\n");

  const double macs =
      static_cast<double>(iters) * kTiles * kC1 * kN * nrows * kK;
  return 2.0 * macs / dt * 1e-9;
}

}  // namespace

int main(int argc, char** argv) {
  const int64_t iters = (argc > 1) ? std::atoll(argv[1]) : 2000;
  std::printf("NROWS=1  %8.1f GFLOP/s\n", run(1, iters));
  std::printf("NROWS=2  %8.1f GFLOP/s\n", run(2, iters));
  return 0;
}
