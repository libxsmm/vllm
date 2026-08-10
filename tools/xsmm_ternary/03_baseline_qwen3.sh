#!/bin/bash
# Baseline: Qwen3-8B BF16 offline generation on the ARL node.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
set -x

export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-8}
export VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-0-23}
export OMP_NUM_THREADS=24

MODEL=${MODEL:-Qwen/Qwen3-8B}
DTYPE=${DTYPE:-bfloat16}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "$VLLM_SRC"
python /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/run_generate.py \
    --model "$MODEL" --dtype "$DTYPE" $EXTRA_ARGS
