// Quick CPUID probe: does this CPU expose AVX-VNNI, AVX-VNNI-INT8 and AVX-NE-CONVERT?
// LIBXSMM requires all of them to select LIBXSMM_X86_AVX2_SRF (the int2xint8 microkernel target).
#include <stdio.h>
#include <cpuid.h>

int main(void) {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
  printf("CPUID.7.1 eax=%08x ebx=%08x ecx=%08x edx=%08x\n", eax, ebx, ecx, edx);
  printf("AVX_VNNI        (eax bit4)  : %d\n", (eax >> 4) & 1);
  printf("AVX_NE_CONVERT  (edx bit5)  : %d\n", (edx >> 5) & 1);
  printf("AVX_VNNI_INT8   (edx bit4)  : %d\n", (edx >> 4) & 1);
  printf("AVX_VNNI_INT16  (edx bit10) : %d\n", (edx >> 10) & 1);
  printf("=> libxsmm arch: %s\n",
         (((eax >> 4) & 1) && (((edx >> 4) & 3) == 3)) ? "srf (int2 kernel OK)"
         : (((eax >> 4) & 1) ? "adl (NO int2 kernel!)" : "avx2"));
  return 0;
}
