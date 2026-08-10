#!/bin/bash
# Create the conda environment for the ternary vLLM CPU build.
set -x
CONDA_ROOT=/data/nfs_home/egeorgan/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda env list
if ! conda env list | grep -q "^ternary_vllm "; then
  conda create -y -n ternary_vllm python=3.12
fi
conda activate ternary_vllm
python -V
which python pip
