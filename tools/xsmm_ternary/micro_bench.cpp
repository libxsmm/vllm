// Standalone timing harness for the ternary microkernel, so compiler/codegen
// experiments do not need a full vLLM rebuild.
//
//   g++  -O3 -march=native -std=c++17 micro_bench.cpp -o mb_gcc
//   icpx -O3 -march=native -std=c++17 micro_bench.cpp -o mb_icx
//
// Mimics the real call pattern: C1 groups of K elements, 32 output columns per
// tile, NROWS activation rows per microkernel call.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../csrc/cpu/xsmm_ternary_micro.hpp"

using namespace vllm_ternary;

namespace {

constexpr int64_t kK = 128;      // group size
constexpr int64_t kC1 = 32;      // groups per panel -> C = 4096
constexpr int64_t kTiles = 64;   // distinct 32-col tiles, keeps data in L2

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <int NROWS>
double run(int64_t iters) {
  const int64_t C = kC1 * kK;
  const int64_t group_bytes = (kK / kMicroVnni) * kMicroN;  // 1024
  const int64_t panel_bytes = kC1 * group_bytes;

  std::vector<uint8_t> w(kTiles * panel_bytes);
  std::vector<int8_t> b(NROWS * C);
  std::vector<int32_t> acc(NROWS * kMicroN * kTiles, 0);

  for (size_t i = 0; i < w.size(); ++i) w[i] = static_cast<uint8_t>(i * 37u);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>((i % 251) - 125);

  const TernaryLuts luts = make_ternary_luts();

  // warmup
  for (int64_t t = 0; t < kTiles; ++t) {
    for (int64_t c1 = 0; c1 < kC1; ++c1) {
      ternary_micro<NROWS>(w.data() + t * panel_bytes + c1 * group_bytes,
                           b.data() + c1 * kK, C,
                           acc.data() + t * NROWS * kMicroN, kK, luts);
    }
  }

  const double t0 = now_s();
  for (int64_t it = 0; it < iters; ++it) {
    for (int64_t t = 0; t < kTiles; ++t) {
      for (int64_t c1 = 0; c1 < kC1; ++c1) {
        ternary_micro<NROWS>(w.data() + t * panel_bytes + c1 * group_bytes,
                             b.data() + c1 * kK, C,
                             acc.data() + t * NROWS * kMicroN, kK, luts);
      }
    }
  }
  const double dt = now_s() - t0;

  // keep the accumulators alive
  int64_t sink = 0;
  for (size_t i = 0; i < acc.size(); ++i) sink += acc[i];
  if (sink == 0x7fffffffffffffffLL) std::fprintf(stderr, "sink\n");

  const double macs =
      static_cast<double>(iters) * kTiles * kC1 * kMicroN * NROWS * kK;
  return 2.0 * macs / dt * 1e-9;  // GFLOP/s
}

}  // namespace

int main(int argc, char** argv) {
  const int64_t iters = (argc > 1) ? std::atoll(argv[1]) : 2000;
  std::printf("NROWS=1  %8.1f GFLOP/s\n", run<1>(iters));
  std::printf("NROWS=2  %8.1f GFLOP/s\n", run<2>(iters));
  return 0;
}
