#!/bin/bash
# Correctness sweep for the LIBXSMM ternary GEMM across shapes/dtypes.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh

TEST=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/test_ternary_gemm.py
cd "$VLLM_SRC"

fail=0
run() {
    echo "### $*"
    if ! python "$TEST" "$@" 2>&1 | grep -E "libxsmm|packed:|max rel|PASS|MISMATCH|Error|\[bench\]"; then
        fail=1
    fi
    echo
}

# GEMV / decode path
run -m 1   -n 256   -k 512  --ternary-weights
run -m 1   -n 4096  -k 4096 --ternary-weights
# reduction-group edge cases: C1==1 (first group is also the last, so the
# accumulator is never pre-written), C1==2 (first+last, no middle group)
run -m 1   -n 256   -k 128  --ternary-weights
run -m 32  -n 256   -k 128  --ternary-weights
run -m 1   -n 256   -k 256  --ternary-weights
run -m 17  -n 256   -k 256  --ternary-weights
run -m 17  -n 256   -k 128  --group-size 128 --ternary-weights --dtype float32
run -m 17  -n 256   -k 64   --group-size 64  --ternary-weights
# exact block, remainder, multi-block
run -m 32  -n 256   -k 512  --ternary-weights
run -m 33  -n 256   -k 512  --ternary-weights
run -m 100 -n 256   -k 512  --ternary-weights
# dtypes
run -m 17  -n 256   -k 512  --ternary-weights --dtype float16
run -m 17  -n 256   -k 512  --ternary-weights --dtype float32
# non-ternary (lossy) weights still must match the reference exactly
run -m 17  -n 256   -k 512
# group size 64
run -m 17  -n 256   -k 512  --group-size 64 --ternary-weights
# real Qwen3-8B / Bonsai-8B shapes
run -m 1   -n 4096  -k 4096 --ternary-weights --bench
run -m 1   -n 12288 -k 4096 --ternary-weights --bench
run -m 128 -n 4096  -k 4096 --ternary-weights --bench
run -m 128 -n 12288 -k 4096 --ternary-weights --bench

exit $fail
