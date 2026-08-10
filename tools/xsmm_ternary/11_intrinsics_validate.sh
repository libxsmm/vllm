#!/bin/bash
# Validation for the pure-intrinsics ternary GEMM (no LIBXSMM dependency):
# correctness sweep, DRAM-streaming GEMV, prefill-like GEMM, and e2e decode.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
cd "$VLLM_SRC"

echo "=== build info ==="
python -c "
import torch, vllm._custom_ops, re
print('  ternary isa :', torch.ops._C.xsmm_target_arch())
print('  loaded ext  :', sorted(set(re.findall(r'_C[A-Za-z0-9_]*\.abi3\.so',
      open('/proc/self/maps').read()))))"

echo
echo "################ CORRECTNESS ################"
bash "$S/04_test_ternary_sweep.sh" 2>&1 \
    | grep -E "^###|max rel|PASS|MISMATCH" | paste - -

echo
echo "################ STANDALONE GEMV (DRAM-streaming, m=1) ################"
printf "%-24s %s\n" "shape" "result"
for SHAPE in "6144 4096" "4096 4096" "24576 4096" "4096 12288" "151680 4096"; do
    set -- $SHAPE
    printf "n=%-6s k=%-6s        " "$1" "$2"
    python "$S/bench_gemv_bw.py" -m 1 -n "$1" -k "$2" \
        --working-set-mb 768 --sweeps 3 2>/dev/null \
        | tail -1 | sed 's/^ *sweep [0-9]*: *//'
done

echo
echo "################ GEMM, m>1 (prefill-like) ################"
for M in 2 8 32; do
    printf "m=%-4s n=4096 k=4096   " "$M"
    python "$S/bench_gemv_bw.py" -m "$M" -n 4096 -k 4096 \
        --working-set-mb 768 --sweeps 3 2>/dev/null \
        | tail -1 | sed 's/^ *sweep [0-9]*: *//'
done

echo
echo "################ E2E DECODE (Bonsai-8B, BS=1, slope method) ################"
MODEL=/data/nfs_home/egeorgan/cpu_ternary_vllm/models/Ternary-Bonsai-8B-xsmm-g128-fp16s
export VLLM_ENABLE_V1_MULTIPROCESSING=0
unset VLLM_CPU_KVCACHE_SPACE
python "$S/bench_decode_clean.py" \
    --model "$MODEL" --dtype bfloat16 --max-model-len 2048 2>/dev/null | tail -2
