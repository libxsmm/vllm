#!/bin/bash
# Final run: correctness sweep + standalone DRAM GEMV + e2e decode at BS=1,
# with all optimizations enabled (in-process engine, no prefix caching,
# allocation-free GEMM path).
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24
export VLLM_ENABLE_V1_MULTIPROCESSING=0

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
MODEL=/data/nfs_home/egeorgan/cpu_ternary_vllm/models/Ternary-Bonsai-8B-xsmm-g128
cd "$VLLM_SRC"

echo "################## CORRECTNESS ##################"
bash "$S/04_test_ternary_sweep.sh"

echo
echo "################## STANDALONE GEMV (DRAM) ##################"
for SHAPE in "6144 4096" "4096 4096" "24576 4096" "4096 12288"; do
    set -- $SHAPE
    echo "--- n=$1 k=$2 (m=1) ---"
    python "$S/bench_gemv_bw.py" -m 1 -n "$1" -k "$2" --working-set-mb 768 --sweeps 3
done

echo
echo "################## E2E DECODE, BS=1 ##################"
python "$S/bench_decode.py" --model "$MODEL" --batch 1 --max-tokens 64

echo
echo "################## E2E SANITY (text output) ##################"
python "$S/run_generate.py" --model "$MODEL" --dtype bfloat16 \
    --max-model-len 1024 --max-tokens 48 --num-prompts 3
