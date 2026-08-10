#!/bin/bash
# 16-bit vs 2-bit head-to-head on a clean node: identical prompt, identical
# sampling, identical 256 output tokens.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_KVCACHE_SPACE=4 VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24
export VLLM_ENABLE_V1_MULTIPROCESSING=0

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
BF16=/data/nfs_home/egeorgan/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-8B-unpacked/snapshots/ac20f03fc62e872399218b659c8e949dfca05769
TERNARY=/data/nfs_home/egeorgan/cpu_ternary_vllm/models/Ternary-Bonsai-8B-xsmm-g128
PROMPT="Tell me about CPU caches in 300 words"

cd "$VLLM_SRC"
python -c "import vllm._custom_ops" || exit 1

for CFG in "BF16_16bit $BF16" "TERNARY_2bit $TERNARY"; do
    set -- $CFG
    echo "################################################################"
    echo "##  $1"
    echo "################################################################"
    free -g | head -2
    echo "--- steady-state decode (slope, prefill-free) ---"
    python "$S/bench_decode_clean.py" --model "$2" --n1 16 --n2 80 --reps 3 \
        2>/dev/null | grep -E "step time|ternary GEMMs|everything else|SINGLE-STREAM|spread"
    echo "--- generation: 256 tokens ---"
    python "$S/run_generate.py" --model "$2" --dtype bfloat16 \
        --max-model-len 2048 --max-tokens 256 --chat --prompt "$PROMPT" \
        2>/dev/null | grep -E "\[timing\]|PROMPT:|OUTPUT:" -A 2
    echo
done
