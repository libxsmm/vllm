#!/bin/bash
# Build vLLM for CPU in editable (develop) mode on the ARL node.
set -eo pipefail
TERNARY_NO_IOMP=1
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
unset HF_HUB_OFFLINE
set -x

cd "$VLLM_SRC"

# a PyPI vllm can be pulled in transitively; remove it so our source build is the only one
python -m pip uninstall -y vllm 2>/dev/null || true

export VLLM_TARGET_DEVICE=cpu
export MAX_JOBS=${MAX_JOBS:-24}
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

python setup.py develop

python -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)"
# _C is the AVX512+AMX variant; on AVX2-only hosts vLLM loads _C_AVX2 at runtime.
python -c "import vllm._custom_ops as ops; import torch; print('custom ops OK:', torch.ops._C.__module__)"
