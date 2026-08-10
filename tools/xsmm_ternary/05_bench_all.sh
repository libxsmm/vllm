#!/bin/bash
# Reports (1) standalone DRAM-streaming GEMV bandwidth and (2) the bandwidth the
# same kernel achieves on the GEMVs inside a real e2e decode run.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
MODEL=/data/nfs_home/egeorgan/cpu_ternary_vllm/models/Ternary-Bonsai-8B-xsmm-g128
cd "$VLLM_SRC"

echo "=================== 1) STANDALONE GEMV (streams from DRAM) ==================="
# Bonsai-8B projection shapes, m=1 (decode)
for SHAPE in "4096 4096" "12288 4096" "4096 12288" "1024 4096"; do
    set -- $SHAPE
    echo "--- n=$1 k=$2 ---"
    python "$S/bench_gemv_bw.py" -m 1 -n "$1" -k "$2" --working-set-mb 768 --sweeps 3
done

echo
echo "=================== 2) GEMVs PROFILED INSIDE E2E DECODE ==================="
for B in 1 4; do
    echo "--- batch=$B ---"
    python "$S/bench_decode.py" --model "$MODEL" --batch "$B" --max-tokens 64
done
