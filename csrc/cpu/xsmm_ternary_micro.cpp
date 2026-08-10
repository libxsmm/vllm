#include "xsmm_ternary_micro.hpp"

namespace vllm_ternary {

// VPDPBSSD and the ymm shuffles need AVX2 + AVX-VNNI-INT8; the same feature set
// LIBXSMM calls "srf".
bool ternary_intrinsics_supported() {
#if defined(XSMM_TERNARY_HAVE_X86) && defined(__GNUC__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnniint8");
#else
  return false;
#endif
}

}  // namespace vllm_ternary
