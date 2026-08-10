#!/bin/bash
# Install build + runtime deps for the vLLM CPU build.
set -eo pipefail
source /data/nfs_home/egeorgan/cpu_ternary_vllm/scripts/env.sh
unset HF_HUB_OFFLINE
set -x

cd "$VLLM_SRC"

conda install -y -n "$CONDA_ENV" pip

python -m pip install --upgrade pip
python -m pip install -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/cpu.txt      --extra-index-url https://download.pytorch.org/whl/cpu

python -c "import torch; print('torch', torch.__version__, torch.__file__)"
