#!/bin/bash
# Final report: standalone GEMVs for every Bonsai-8B shape, steady-state e2e
# decode, and a real generation sample.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24
export VLLM_ENABLE_V1_MULTIPROCESSING=0

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
MODEL=/data/nfs_home/egeorgan/cpu_ternary_vllm/models/Ternary-Bonsai-8B-xsmm-g128
cd "$VLLM_SRC"

python -c "import vllm._custom_ops" || exit 1

echo "################ 1) STANDALONE GEMVs (DRAM-streaming, m=1) ################"
echo "shape                        us/call     GB/s     GFLOP/s"
for SHAPE in "6144 4096 qkv_proj" "4096 4096 o_proj" "24576 4096 gate_up_proj" \
             "4096 12288 down_proj" "151680 4096 lm_head"; do
    set -- $SHAPE
    printf "%-14s n=%-6s k=%-6s " "$3" "$1" "$2"
    python "$S/bench_gemv_bw.py" -m 1 -n "$1" -k "$2" \
        --working-set-mb 768 --sweeps 3 2>/dev/null \
        | tail -1 | sed 's/^ *sweep [0-9]*: *//'
done

echo
echo "################ 2) E2E STEADY-STATE DECODE (BS=1) ################"
python "$S/bench_decode_clean.py" --model "$MODEL" --n1 16 --n2 80 --reps 3

echo
echo "################ 3) GENERATION SAMPLE ################"
python "$S/run_generate.py" --model "$MODEL" --dtype bfloat16 \
    --max-model-len 2048 --max-tokens 256 \
    --prompt "Tell me about CPU caches in 300 words"
