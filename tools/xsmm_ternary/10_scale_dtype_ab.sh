#!/bin/bash
# A/B the weight-scale precision: fp32 vs fp16 scales.
# Scales are 1 per `group_size` values, i.e. 11% of weight bytes at fp32.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
export VLLM_CPU_OMP_THREADS_BIND=0-23 OMP_NUM_THREADS=24

S=/data/nfs_home/egeorgan/cpu_ternary_vllm/scripts
cd "$VLLM_SRC"
python -c "import vllm._custom_ops" || exit 1

echo "################ CORRECTNESS ################"
for FLAG in "--fp32-scales" ""; do
    LBL=$([ -n "$FLAG" ] && echo "fp32 scales" || echo "fp16 scales")
    for CFG in "-m 1 -n 4096 -k 4096" "-m 17 -n 256 -k 512 --dtype float32" \
               "-m 1 -n 256 -k 128" "-m 128 -n 4096 -k 4096"; do
        printf "%-12s %-38s " "$LBL" "$CFG"
        python "$S/test_ternary_gemm.py" $CFG --ternary-weights $FLAG 2>/dev/null \
            | grep -E "max rel|PASS|MISMATCH" | tr '\n' ' '
        echo
    done
done

echo
echo "################ STANDALONE GEMV: fp32 vs fp16 scales ################"
printf "%-14s %-22s %s\n" "shape" "scales" "result"
for SHAPE in "6144 4096" "24576 4096" "4096 12288" "151680 4096"; do
    set -- $SHAPE
    for FLAG in "--fp32-scales" ""; do
        LBL=$([ -n "$FLAG" ] && echo "fp32" || echo "fp16")
        printf "n=%-6s k=%-6s  %-6s " "$1" "$2" "$LBL"
        python "$S/bench_gemv_bw.py" -m 1 -n "$1" -k "$2" \
            --working-set-mb 768 --sweeps 3 $FLAG 2>/dev/null \
            | tail -1 | sed 's/^ *sweep [0-9]*: *//'
    done
done
