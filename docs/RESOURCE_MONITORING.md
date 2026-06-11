# Resource Monitoring and Results Structure

## Overview

This document describes the resource monitoring implementation and the new results directory structure that separates baseline paper reproduction from improved implementations.

## Resource Monitoring

### What is Monitored

All operations now monitor CPU, Memory, and GPU usage automatically:

1. **Dataset Download**: Network I/O and disk operations
2. **Dataset Preprocessing**: CPU-intensive TF-IDF, text cleaning
3. **Model Training**: Training time and peak memory per model
4. **Model Evaluation**: Inference time and memory usage

### Monitoring Data Structure

Each monitored operation captures:

```python
{
    "operation": "operation_name",
    "start_time": "2025-06-11T10:30:00",
    "end_time": "2025-06-11T10:35:00",
    "elapsed_seconds": 300.5,
    
    "start_resources": {
        "cpu_percent": 15.2,
        "memory_used_gb": 2.4,
        "memory_available_gb": 5.6,
        "memory_percent": 30.0,
        "memory_total_gb": 8.0,
        "gpu_used_gb": 0.0,  # If GPU available
        "gpu_reserved_gb": 0.0,
        "gpu_total_gb": 8.0,
        "gpu_percent": 0.0
    },
    
    "end_resources": {
        "cpu_percent": 85.3,
        "memory_used_gb": 6.8,
        "memory_available_gb": 1.2,
        "memory_percent": 85.0,
        "memory_total_gb": 8.0,
        "gpu_used_gb": 4.2,  # If GPU used
        "gpu_reserved_gb": 4.5,
        "gpu_total_gb": 8.0,
        "gpu_percent": 52.5
    },
    
    "resource_delta": {
        "cpu_percent": 70.1,
        "memory_used_gb": 4.4,
        "memory_percent": 55.0,
        "gpu_used_gb": 4.2,
        "gpu_percent": 52.5
    }
}
```

## Results Directory Structure

The new structure separates results into three main categories:

```
.artifacts/results/
├── baseline/                    # Paper baseline reproduction (H0)
│   ├── 20250611_103000/         # Timestamped run
│   │   ├── metrics.csv
│   │   ├── metrics.json
│   │   ├── metadata.json
│   │   ├── resource_monitoring.json  # Complete resource data
│   │   ├── summary.md
│   │   └── command.txt
│   └── 20250611_153000/         # Another run
│
├── implementacao/               # Code improvements and optimizations
│   ├── 20250611_120000/         # Timestamped run
│   │   ├── metrics.csv
│   │   ├── metadata.json
│   │   ├── resource_monitoring.json
│   │   └── ...
│
├── runs/                        # Research experiments (H1-H6)
│   ├── h1_meajor/
│   │   └── 20250611_140000/
│   ├── h2_url_meta/
│   ├── h3_fusion/
│   ├── h4_compact_encoders/
│   ├── h5_robustness/
│   └── h6_base_rates/
│
└── research/                    # Aggregate research tables
    ├── all_experiments_metrics.csv  # Combined H1-H6 results
    └── summaries/
        └── experiment_name.csv
```

## When to Use Each Function

### Baseline Paper Reproduction (H0)

**Command**: `python3 run.py train`

**Function**: `save_baseline_run()`

**Purpose**: Reproduce the exact results from the IEEE WF-PST 2025 paper

**Location**: `.artifacts/results/baseline/<timestamp>/`

**Use when**:
- Running the original paper implementation
- Reproducing baseline metrics for comparison
- Verifying paper claims

### Improved Implementations

**Function**: `save_implementacao_run()`

**Purpose**: Track code improvements and optimizations over time

**Location**: `.artifacts/results/implementacao/<timestamp>/`

**Use when**:
- Improving the baseline code (e.g., better preprocessing, optimized training)
- Testing new optimization techniques
- Comparing improved versions against baseline
- NOT testing new research hypotheses (use research experiments instead)

### Research Experiments (H1-H6)

**Function**: `save_experiment_run()`

**Purpose**: Test scientific hypotheses about phishing detection

**Location**: `.artifacts/results/runs/h<experiment_name>/<timestamp>/`

**Use when**:
- Testing H1-H6 research hypotheses
- Running orchestrated experiment suites (paper-v1, smoke, confirmatory)
- Comparing different datasets, splits, or model configurations

## Resource Monitoring in Results Files

### Files That Include Resource Data

1. **resource_monitoring.json**: Complete raw resource data for all operations
2. **metadata.json**: Includes high-level resource summary (training times, peak memory)
3. **results_summary.txt**: Human-readable resource monitoring section
4. **model_results.json**: Includes per-model resource metrics

### Resource Data in Text Reports

The `results_summary.txt` now includes a dedicated section:

```
================================================================================
RESOURCE MONITORING DETAILS
================================================================================

Operation: DATASET_DOWNLOAD
--------------------------------------------------------------------------------
Elapsed Time: 45.30 seconds
Start Time: 2025-06-11T10:30:00
End Time: 2025-06-11T10:30:45

CPU Usage:
  Start: 5.2%
  End:   15.8%

Memory Usage:
  Start: 2.40 GB (30.0%)
  End:   3.20 GB (40.0%)
  Total: 8.00 GB
  Available: 4.80 GB

Operation: NAIVE_BAYES_TRAINING
--------------------------------------------------------------------------------
Elapsed Time: 12.50 seconds
...
```

## Implementation Details

### Modified Files

1. **src/phishing_detection/utils/resource_monitor.py**
   - Context manager for monitoring operations
   - Captures CPU, memory, and GPU metrics
   - Calculates resource deltas

2. **src/phishing_detection/experiments/results.py**
   - Added `resource_logs` parameter to `save_baseline_run()`
   - Added `resource_logs` parameter to `save_experiment_run()`
   - Added new `save_implementacao_run()` function
   - All functions now save `resource_monitoring.json`

3. **src/phishing_detection/main.py**
   - `save_results()` passes `resource_logs` to exporters
   - `save_registry_run()` includes `resource_logs` in baseline save
   - `save_run_metadata()` includes complete `resource_logs`

4. **src/phishing_detection/utils/results_exporter.py**
   - `export_results_to_text()` includes resource monitoring section
   - `export_results_to_json()` includes complete resource data
   - `export_all_formats()` accepts `resource_logs` parameter

## Usage Examples

### Baseline Reproduction with Resource Monitoring

```bash
python3 run.py train
```

Output:
- `.artifacts/results/baseline/20250611_103000/`
  - All metrics, metadata, and complete resource data

### Improved Implementation with Resource Monitoring

```python
from phishing_detection.experiments.results import save_implementacao_run

# Run improved code
results = [...]
metadata = {...}
resource_logs = pipeline.resource_logs

save_implementacao_run(
    experiment_name="optimized_v1",
    results=results,
    metadata=metadata,
    command="python3 run_improved.py",
    summary_title="Optimized Implementation v1",
    resource_logs=resource_logs
)
```

Output:
- `.artifacts/results/implementacao/20250611_120000/`
  - All metrics, metadata, and complete resource data

## Comparing Results Across Runs

### Compare Baseline vs Implementation

```bash
# Baseline
ls .artifacts/results/baseline/

# Implementation
ls .artifacts/results/implementacao/

# Compare resource usage
cat .artifacts/results/baseline/*/resource_monitoring.json
cat .artifacts/results/implementacao/*/resource_monitoring.json
```

### Track Improvements Over Time

```bash
# List all implementations chronologically
ls -lt .artifacts/results/implementacao/

# Compare specific metrics
diff .artifacts/results/baseline/20250611_103000/metrics.json \
     .artifacts/results/implementacao/20250611_120000/metrics.json
```

## Important Notes

1. **Separation of Concerns**: Baseline, implementation, and research results are kept separate to avoid mixing different types of experiments

2. **Resource Data Completeness**: All resource monitoring data is now saved permanently, not just logged to console

3. **Reproducibility**: Each run includes git commit, hardware metadata, and complete resource logs

4. **No Data Loss**: Resource logs are no longer lost after program execution - they're saved to JSON files

5. **Aggregate Research**: Only research experiments (H1-H6) are added to `all_experiments_metrics.csv`. Baseline and implementation runs stay in their timestamped directories.

## Future Improvements

1. Add resource usage visualization graphs
2. Implement automatic comparison between baseline and implementation runs
3. Create resource efficiency metrics (e.g., accuracy per second, accuracy per GB)
4. Add cost estimation based on cloud resource pricing
