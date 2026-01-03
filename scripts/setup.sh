#!/bin/bash
# Setup script for mine.cu
# Configures CUDA 12.8 and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check for CUDA 12.8
if [ ! -d "/usr/local/cuda-12.8" ]; then
    echo "Error: CUDA 12.8 not found at /usr/local/cuda-12.8"
    echo "Install CUDA 12.8 from: https://developer.nvidia.com/cuda-12-8-0-download-archive"
    exit 1
fi

echo "Using CUDA 12.8 at /usr/local/cuda-12.8"

cd "$PROJECT_DIR"

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python 3.12

# Install PyTorch with CUDA 12.8
echo "Installing PyTorch with CUDA 12.8..."
uv pip install torch numpy --index-url https://download.pytorch.org/whl/cu128

# Build and install mine.cu
echo "Building mine.cu..."
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH uv pip install -e .

echo ""
echo "Setup complete!"
echo "Activate with: source .venv/bin/activate"
echo "Test with: python -c 'from minecu import MineEnv; print(\"OK\")'"
