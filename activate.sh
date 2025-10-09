#!/bin/bash
# Activate virtual environment and set environment variables for this project

# Get script directory (works with source command)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# --- CUDA 12.8 Path setup ---
CUDA_VERSION="12.8"
CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"

if [ -d "$CUDA_PATH" ]; then
    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
    echo "✅ Using CUDA ${CUDA_VERSION} at ${CUDA_PATH}"
else
    echo "⚠️ CUDA ${CUDA_VERSION} not found at ${CUDA_PATH}"
fi

# Activate Python virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# Set Blackwell (RTX 50XX) compatibility
export TORCH_CUDA_ARCH_LIST="12.0"

# Set Hugging Face cache to local directory
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${SCRIPT_DIR}/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="${SCRIPT_DIR}/.cache/huggingface/transformers"

echo "✅ Virtual environment activated"
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "nvcc: $(command -v nvcc 2>/dev/null || echo 'not found')"
