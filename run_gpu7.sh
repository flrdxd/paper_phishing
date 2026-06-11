#!/bin/bash
# Script to run phishing detection experiments on GPU 7 only
# Usage: ./run_gpu7.sh [command_args]
#
# Examples:
#   ./run_gpu7.sh train                    # Baseline reproduction on GPU 7
#   ./run_gpu7.sh experiment h1-meajor     # Experiment H1 on GPU 7
#   ./run_gpu7.sh experiment paper-v1      # Full suite on GPU 7

# Set CUDA to use only GPU 7
# This maps physical GPU 7 to local GPU 0 in PyTorch
export CUDA_VISIBLE_DEVICES=7

# Log which GPU we're using
echo "=========================================="
echo "Running on GPU 7 only (mapped to local GPU 0)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Verify GPU is available
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

echo ""
echo "=========================================="
echo "Starting experiment..."
echo "=========================================="
echo ""

# Run the command with all arguments
python3 run.py "$@"
