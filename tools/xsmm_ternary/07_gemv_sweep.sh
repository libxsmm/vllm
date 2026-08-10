#!/bin/bash
# Standalone DRAM-streaming GEMV sweep across every shape the model actually
# uses, including lm_head. Used to compare scalar vs AVX2 quant/dequant.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
cd "$VLLM_SRC"

echo "shape                          us/call     GB/s    GFLOP/s"
# qkv_proj, o_proj, gate_up_proj, down_proj, lm_head
for SHAPE in "6144 4096" "4096 4096" "24576 4096" "4096 12288" "151680 4096"; do
    set -- $SHAPE
    printf "n=%-6s k=%-6s  " "$1" "$2"
    python "$S/bench_gemv_bw.py" -m 1 -n "$1" -k "$2" \
        --working-set-mb 768 --sweeps 3 2>/dev/null \
        | tail -1 | sed 's/^ *sweep [0-9]*: *//'
done
