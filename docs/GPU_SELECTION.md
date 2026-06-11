# GPU Selection Guide

## Overview

This project now supports running experiments on specific GPUs in multi-GPU servers. This is useful for:
- Sharing GPU resources among multiple users/projects
- Running experiments on a specific GPU while others are in use
- Monitoring resource usage per GPU

## Quick Start: GPU 7 Only

### Method 1: Using the convenience script (Recommended)

```bash
# Baseline reproduction on GPU 7
./run_gpu7.sh train

# Research experiments on GPU 7
./run_gpu7.sh experiment h1-meajor --split source
./run_gpu7.sh experiment paper-v1 --profile smoke

# Your own commands
./run_gpu7.sh check
./run_gpu7.sh test
```

### Method 2: Manual CUDA_VISIBLE_DEVICES

```bash
# Set environment variable
export CUDA_VISIBLE_DEVICES=7

# Run normally
python3 run.py train
python3 run.py experiment h1-meajor
```

### Method 3: One-line command

```bash
# Single command with GPU selection
CUDA_VISIBLE_DEVICES=7 python3 run.py train

# Experiment with GPU selection
CUDA_VISIBLE_DEVICES=7 python3 run.py experiment h1-meajor --split source
```

## How It Works

### CUDA_VISIBLE_DEVICES Behavior

When you set `CUDA_VISIBLE_DEVICES=7`:
- PyTorch sees only 1 GPU (device_count = 1)
- Physical GPU 7 becomes local GPU 0 in PyTorch
- All torch.cuda calls automatically use GPU 7
- No code changes needed!

### Example Mapping

```python
import torch

# Without CUDA_VISIBLE_DEVICES
# torch.cuda.device_count() = 8
# torch.cuda.get_device_name(0) = "NVIDIA A100 80GB PCIe" (physical GPU 0)
# torch.cuda.get_device_name(7) = "NVIDIA A100 80GB PCIe" (physical GPU 7)

# With CUDA_VISIBLE_DEVICES=7
# torch.cuda.device_count() = 1
# torch.cuda.get_device_name(0) = "NVIDIA A100 80GB PCIe" (physical GPU 7)
# torch.cuda.get_device_name(1) = ERROR (out of bounds)
```

## Resource Monitoring with GPU Selection

### What Gets Monitored

The resource monitoring system automatically adapts to your GPU selection:

```json
{
  "gpu_used_gb": 4.2,
  "gpu_reserved_gb": 4.5,
  "gpu_total_gb": 80.0,
  "gpu_percent": 5.25,
  "gpu_id": 0,                      // Local ID (always 0 after filtering)
  "gpu_name": "NVIDIA A100 80GB PCIe",
  "gpu_count_visible": 1,
  "cuda_visible_devices": "7",      // Your selection
  "original_gpu_mapping": "Local GPU 0 → Physical GPU 7"
}
```

### Verification

Before running experiments, verify GPU selection:

```bash
# Check which GPUs PyTorch sees
python3 -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# With CUDA_VISIBLE_DEVICES=7
CUDA_VISIBLE_DEVICES=7 python3 -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Using Different GPUs

### GPU 0 (First GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py train
# OR
./run_gpu0.sh  # If you create this script
```

### GPU 3 (Fourth GPU)

```bash
CUDA_VISIBLE_DEVICES=3 python3 run.py train
# OR
export CUDA_VISIBLE_DEVICES=3
python3 run.py train
```

### Multiple GPUs

If you want to use GPUs 5, 6, and 7 together:

```bash
CUDA_VISIBLE_DEVICES=5,6,7 python3 run.py train
# PyTorch sees 3 GPUs: local 0, 1, 2 (physical 5, 6, 7)
```

## Current Server Status

Based on your nvidia-smi output:

| GPU | Memory Used | Status | Recommended Use |
|-----|-------------|--------|-----------------|
| 0 | 4.5 GB / 80 GB | Lightly used | Available |
| 1 | 0.9 GB / 80 GB | Nearly free | Available |
| 2 | 1.7 GB / 80 GB | Lightly used | Available |
| 3 | 0.0 GB / 80 GB | **FREE** | ✅ Best choice |
| 4 | 29.0 GB / 80 GB | Heavily used | Avoid |
| 5 | 63.2 GB / 80 GB | Heavily used | Avoid |
| 6 | 52.8 GB / 80 GB | Heavily used | Avoid |
| 7 | 64.7 GB / 80 GB | Heavily used | ⚠️ Your choice |

**Recommendation**: GPU 3 is completely free and would be faster than GPU 7.

## Switch to GPU 3 (Better Option)

If you want to use the free GPU 3 instead:

```bash
# Create script for GPU 3
cat > run_gpu3.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
echo "Running on GPU 3 only"
python3 run.py "$@"
EOF

chmod +x run_gpu3.sh

# Use it
./run_gpu3.sh train
```

Or modify the existing script:

```bash
# Edit run_gpu7.sh and change line 7:
# export CUDA_VISIBLE_DEVICES=7
# to:
# export CUDA_VISIBLE_DEVICES=3
```

## Checking GPU Usage During Training

### In another terminal (monitor GPU 7):

```bash
# Watch GPU 7 usage in real-time
watch -n 1 nvidia-smi

# Or monitor only GPU 7
watch -n 1 "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | grep '^7,'"

# With highlighted GPU 7
watch -n 1 "nvidia-smi && echo '=== GPU 7 ===' && nvidia-smi -i 7"
```

### From Python (during experiment):

```python
import torch
import psutil

print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory Peak: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: The GPU is already heavily used (64.7 GB / 80 GB). Try:
1. Use a different GPU (GPU 3 is free)
2. Reduce batch size: `python3 run.py experiment h4-compact-encoders --batch-size 4`
3. Clear GPU cache: Restart your Python process

### Issue: "PyTorch sees 0 GPUs"

**Possible causes**:
1. Wrong GPU ID (check with `nvidia-smi`)
2. CUDA not installed (check with `python3 -c "import torch; print(torch.cuda.is_available())"`)

### Issue: Resource monitoring shows 0 GPU usage

**Possible causes**:
1. Running CPU-only models (Naive Bayes, etc.)
2. GPU selection failed silently
3. Check logs for GPU errors

## Scripts Available

- `run_gpu7.sh` - Run experiments on GPU 7
- `run_gpu3.sh` - Run experiments on GPU 3 (if you create it)
- `run_gpu.sh <ID>` - Generic script (if you create it)

## Best Practices

1. **Check GPU status before starting**: `nvidia-smi`
2. **Use free GPUs when possible**: GPU 3 is currently free
3. **Monitor during training**: `watch -n 1 nvidia-smi`
4. **Clear GPU memory between runs**: Restart Python process
5. **Resource monitoring is automatic**: Check `resource_monitoring.json` after each run

## Example Workflow

```bash
# 1. Check GPU status
nvidia-smi

# 2. Set GPU (example: GPU 7)
export CUDA_VISIBLE_DEVICES=7

# 3. Verify PyTorch sees it
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0)}')"

# 4. Run experiment
python3 run.py train

# 5. Monitor resource usage in another terminal
watch -n 1 nvidia-smi

# 6. Check results
cat .artifacts/results/baseline/*/resource_monitoring.json
```

## Performance Notes

- **GPU 7**: Currently using 64.7 GB / 80 GB (80.9% full)
  - May run slower due to existing load
  - Risk of OOM if experiments need more VRAM

- **GPU 3**: Completely free (0 GB / 80 GB)
  - Faster performance
  - Lower OOM risk
  - **Recommended alternative**

To switch to GPU 3, just change one line in `run_gpu7.sh`:
```bash
export CUDA_VISIBLE_DEVICES=3  # Change from 7 to 3
```

Then rename the script:
```bash
mv run_gpu7.sh run_gpu3.sh
```
