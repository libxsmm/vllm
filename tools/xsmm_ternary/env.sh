#!/bin/bash
# Common environment for the ternary-vLLM CPU work on pcl-arl01 (Arrow Lake, AVX2+AVX_VNNI).
export TERNARY_ROOT=/data/nfs_home/egeorgan/cpu_ternary_vllm
export VLLM_SRC=$TERNARY_ROOT/vllm
export CONDA_ROOT=/data/nfs_home/egeorgan/miniforge3
export CONDA_ENV=ternary_vllm
export SLURM_TARGET_JOBID=355139

# conda's (de)activate.d hooks are not `set -u` clean, so relax it while sourcing
__ternary_saved_opts=$(set +o)
set +u
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $CONDA_ENV
eval "$__ternary_saved_opts"
unset __ternary_saved_opts

export VLLM_TARGET_DEVICE=cpu
export CC=gcc
export CXX=g++
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-24}
export MAX_JOBS=${MAX_JOBS:-24}

# Intel OpenMP: torch ships libgomp, whose barriers are costly when the model
# enters ~144 parallel regions per token. libiomp5 must be preloaded to win.
if [[ -z "${TERNARY_NO_IOMP:-}" && -f "$CONDA_PREFIX/lib/libiomp5.so" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
    export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1}
    export KMP_TPAUSE=0
    export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
    export KMP_PLAIN_BARRIER_PATTERN=dist,dist
    export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
fi

# HF cache already populated by the user
export HF_HOME=/data/nfs_home/egeorgan/.cache/huggingface
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
