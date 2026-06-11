#!/bin/bash
# Script to run phishing detection experiments on a specific GPU
# Usage: ./run_gpu.sh <gpu_id> [command_args]
#
# Examples:
#   ./run_gpu.sh 7 train                    # Baseline on GPU 7
#   ./run_gpu.sh 3 experiment h1-meajor     # Experiment H1 on GPU 3
#   ./run_gpu.sh 0 experiment paper-v1      # Full suite on GPU 0

# Check if GPU ID is provided
if [ -z "$1" ]; then
    echo "Error: GPU ID is required"
    echo "Usage: $0 <gpu_id> [command_args]"
    echo "Example: $0 7 train"
    echo ""
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (Run 'nvidia-smi' to see available GPUs)"
    exit 1
fi

GPU_ID=$1
shift  # Remove GPU ID from arguments, keep rest for run.py

# Validate GPU ID
if ! nvidia-smi -i $GPU_ID &>/dev/null; then
    echo "Error: GPU $GPU_ID not found"
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo "  (Run 'nvidia-smi' to see available GPUs)"
    exit 1
fi

# Set CUDA to use only the specified GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Log which GPU we're using
echo "=========================================="
echo "Running on GPU $GPU_ID only (mapped to local GPU 0)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi -i $GPU_ID --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null
echo ""

# Verify GPU is available to PyTorch
echo "PyTorch GPU Detection:"
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  Current device: {torch.cuda.current_device()}')
    print(f'  Device name: {torch.cuda.get_device_name(0)}')
    print(f'  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
    print(f'  Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB')
" 2>/dev/null

echo ""
echo "=========================================="
echo "Starting experiment on GPU $GPU_ID..."
echo "Command: python3 run.py $@"
echo "=========================================="
echo ""

# Run the command with remaining arguments
python3 run.py "$@"
