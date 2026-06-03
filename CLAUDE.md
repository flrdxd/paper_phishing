# Phishing Detection Research Project

## Project Overview

This is a research project for reproducing and extending a phishing email detection paper. The project consists of two main components:

1. **Baseline Paper Reproduction**: Reproducing the results from `docs/Artigo_Phishing.pdf` comparing Naive Bayes, Dandelion-Naive Bayes, BERT, and DistilBERT models
2. **Research Extension**: Testing new hypotheses about SLM-first, URL-aware phishing detection under realistic conditions

**Critical Context**: This project implements a complete research pipeline following rigorous scientific methodology. All experimental results must be logged, reproducible, and statistically validated before publication.

## Project Philosophy

This project follows the **scientific method** over **confirmatory research**:

- We start with questions, not conclusions
- Experiments are designed to discover which hypotheses are defensible
- The final paper thesis emerges from experimental evidence, not predetermined narratives
- Results must be statistically validated with bootstrap confidence intervals
- All experiments are logged with timestamps, git commits, seeds, and hardware metadata

## Core Thesis Statement

**Proposed Contribution**: A SLM-first hybrid detector combining compact text encoders, URL/metadata features, and calibrated fusion can maintain high phishing recall with lower cost, lower latency, and fewer false positives in realistic scenarios, especially under AI-rewritten phishing attacks and realistic base rates.

**SLM Definition**: In this project, **SLM** means **compact transformer encoders / small language encoders** for local classification (DistilBERT, MiniLM, TinyBERT, MobileBERT, DeBERTa-v3-small), NOT small generative LLMs.

## Project Structure

```
├── run.py                           # Main entry point for all commands
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
├── docs/                           # All documentation
│   ├── Artigo_Phishing.pdf         # Original paper baseline
│   ├── proposta_paper_slm_phishing.md  # Full research proposal (READ THIS)
│   ├── INSTRUCOES_EXECUCAO_PAPER.md   # Operational runbook (READ THIS)
│   └── EXPERIMENTS.md              # Experiment protocol
├── src/phishing_detection/         # Main package
│   ├── experiments/                # Research experiments H1-H6, paper-v1
│   ├── models/                     # Baseline models (NB, BERT, etc.)
│   └── utils/                      # Evaluation, audit, export tools
├── tests/                          # Test suite
└── .artifacts/                     # Generated outputs (gitignored)
```

## Key Research Hypotheses

### H0 - Baseline Reproduction (COMPLETED)
Reproduce Naive Bayes, Dandelion-NB, BERT, and DistilBERT from the original paper.
- **Current Status**: BERT/DistilBERT reproduced with high fidelity, Dandelion-NB did not reproduce the reported gain
- **Command**: `python3 run.py train`

### H1 - MeAJOR Generalization (IMPLEMENTED)
Test if realistic multi-source datasets reduce apparent performance compared to random splits.
- **Key Question**: Does source-held-out evaluation reveal generalization issues?
- **Command**: `python3 run.py experiment h1-meajor --split random` vs `--split source`

### H2 - URL/Metadata Features (IMPLEMENTED)
Test if URL and metadata features improve detection beyond text-only.
- **Key Question**: Do structured features reduce false positives while maintaining recall?
- **Command**: `python3 run.py experiment h2-url-meta --split random`

### H3 - Fusion and Calibration (IMPLEMENTED)
Test calibrated fusion of text and URL/metadata scores.
- **Key Question**: Does fusion reduce FPR@Recall>=98% compared to text-only?
- **Command**: `python3 run.py experiment h3-fusion --split random`

### H4 - Compact Encoders (IMPLEMENTED)
Test if compact encoders match BERT performance with lower cost.
- **Key Question**: Can MiniLM/TinyBERT maintain performance with better latency/memory?
- **Command**: `python3 run.py experiment h4-compact-encoders --models distilbert,minilm`

### H5 - Robustness Testing (IMPLEMENTED)
Test performance against controlled text rewrites simulating AI improvements.
- **Key Question**: How much do models degrade when phishing text is sanitized?
- **Command**: `python3 run.py experiment h5-robustness`

### H6 - Realistic Base Rates (IMPLEMENTED)
Test operational impact under realistic phishing prevalence (0.5-10%).
- **Key Question**: Which model generates the fewest false positives per 10k emails?
- **Command**: `python3 run.py experiment h6-base-rates --split random`

## Experimental Design Principles

### Phase 1 - Exploratory (Current Phase)
- Test hypotheses with smoke samples (1000 samples) for rapid iteration
- Use experiments to discover which arguments are actually defensible
- Accept that some components will be discarded if they don't show clear improvement
- **Decision Criteria**: If a component shows small/unstable gains, remove it from v1

### Phase 2 - Confirmatory (Final Phase)
- Freeze architecture, models, features, thresholds, seeds, and metrics
- Run final test ONCE - no cherry-picking
- Apply statistical validation (bootstrap, McNemar, paired tests)
- Write paper based on these results, not earlier exploratory runs

### Metrics That Matter

**Primary Metric**: `FPR@Recall>=98%`
- Forces comparison under realistic operational constraint
- In email security, missing phishing is unacceptable, but false positives have operational cost

**Secondary Metrics**:
- MCC, AUC-PR for imbalanced evaluation
- FP/10k emails, cost/10k for operational analysis  
- Latency p95, throughput for deployment feasibility
- Robustness gap (F1_original - F1_adversarial)

**Anti-Patterns**:
- Don't optimize only for accuracy on balanced data
- Don't claim deployment readiness based only on random split performance
- Don't report results without confidence intervals

## Data Integrity Safeguards

The project implements automatic data quality checks to prevent artificial accuracy inflation:

- **Template Detection**: Rejects datasets with high template repetition
- **Vocabulary Checks**: Flags abnormally small vocabularies
- **Class Distribution Validation**: Ensures dataset matches paper's reported distribution
- **Deduplication**: Removes exact duplicates across train/test splits
- **Source-Held-Out Splits**: Prevents same-source leakage

**Warning Signs**: If accuracy is exactly 100% or training finishes unrealistically fast, STOP and audit the data in `.artifacts/data/`.

## Required Execution Workflow

### Step 1 - Environment Validation
```bash
python3 run.py check  # Validates Kaggle setup
python3 run.py test   # Runs test suite
```

### Step 2 - Baseline Reproduction
```bash
python3 run.py train  # Reproduce paper baseline
```
**Save Results**: Copy `.artifacts/results/runs/baseline_paper/<timestamp>/` to backup location

### Step 3 - Smoke Test Suite
```bash
python3 run.py experiment paper-v1 --profile smoke
```
This runs H1, H2, H3, H5, H6 with 1000 samples each. Excludes H4 (transformers).

### Step 4 - Individual Experiments (Optional)
Run specific hypotheses with larger samples:
```bash
python3 run.py experiment h1-meajor --models nb,logreg,svm --split leave-one-source-out
python3 run.py experiment h4-compact-encoders --epochs 3
```

### Step 5 - Full Experimental Suite
```bash
python3 run.py experiment paper-v1 --profile full
```
Includes H4 with transformer fine-tuning (requires GPU for reasonable runtime).

### Step 6 - Confirmatory Test
After exploratory analysis and architecture freezing:
```bash
python3 run.py experiment paper-v1 --profile confirmatory
```
This result becomes the final paper result - run only ONCE.

## Results Organization

All experimental outputs are organized in `.artifacts/` (gitignored):

```
.artifacts/
├── data/              # Downloaded datasets (Kaggle, MeAJOR, etc.)
├── models/            # Trained model checkpoints
├── plots/             # Generated figures and visualizations
├── results/           # Metrics and summaries
│   ├── research/      # Aggregate tables and summaries
│   └── runs/          # Per-execution detailed results
│       ├── baseline_paper/<timestamp>/
│       ├── h1_meajor/<timestamp>/
│       ├── h2_url_meta/<timestamp>/
│       └── ...
└── logs/              # Execution logs
```

**Important**: `.artifacts/` is NOT in git. For important results, backup the specific timestamped run folder or copy final CSVs to external storage.

## Current Project Status

### Completed Components
- ✅ Baseline paper reproduction pipeline (H0)
- ✅ MeAJOR multi-source dataset integration
- ✅ All 6 research hypotheses implemented (H1-H6)
- ✅ Paper-v1 orchestrated suite (smoke/full/confirmatory profiles)
- ✅ Data quality audit framework
- ✅ Metrics calculation including FPR@Recall>=98%
- ✅ Source-held-out evaluation for generalization testing
- ✅ URL/metadata feature extraction
- ✅ Calibrated fusion implementation
- ✅ Base rate simulation framework
- ✅ Comprehensive test coverage

### Known Limitations
- ⚠️ Dandelion-NB did not reproduce the paper's reported gain
- ⚠️ No experimental results have been generated yet in current environment
- ⚠️ GPU availability will significantly impact H4 transformer training times

### Next Critical Steps
1. **Setup Kaggle credentials** for baseline reproduction
2. **Run baseline** to establish H0 foundation
3. **Execute smoke test** to validate research pipeline
4. **Generate initial experimental results** for exploratory analysis
5. **Statistical analysis** once exploratory phase reveals promising directions
6. **Confirmatory test** after architecture freezing

## Statistical Requirements

All experimental results must include:

- **Bootstrap 95% confidence intervals** for F1, MCC, AUC-PR, FPR
- **McNemar test** for comparing classifiers on same test set
- **Paired bootstrap** for metric differences
- **Holm-Bonferroni correction** for multiple comparisons
- **Effect size reporting**: absolute difference, relative reduction, cost improvement

**Example Conclusion Format**:
> "The hybrid model reduced FPR by 38% (95% CI: 31-45%) compared to text-only while maintaining recall >=98%, with 2.4x lower latency p95 than BERT."

## Reference Materials

### Must-Read Documents (Priority Order)
1. **docs/proposta_paper_slm_phishing.md** - Complete research proposal with all hypotheses
2. **docs/INSTRUCOES_EXECUCAO_PAPER.md** - Operational runbook with commands and workflows
3. **docs/EXPERIMENTS.md** - Experiment protocol and execution order
4. **docs/README_AUDIT.md** - Data quality safeguards

### External References (from proposal)
- Baseline Paper: IEEE WF-PST 2025
- MeAJOR Corpus: https://arxiv.org/abs/2507.17978
- PhiUSIIL URL Dataset: https://archive.ics.uci.edu/dataset/967
- E-PhishGen: https://arxiv.org/abs/2509.01791
- Related Work: MultiPhishGuard, ChatSpamDetector, LLM-PEA, PiMRef

## Development Commands

### Validation and Testing
```bash
python3 run.py check    # Verify Kaggle setup and project structure
python3 run.py test     # Run full test suite
```

### Training and Experiments
```bash
# Baseline reproduction
python3 run.py train

# Individual experiments
python3 run.py experiment h1-meajor --sample-size 1000 --models nb,logreg --split random
python3 run.py experiment h2-url-meta --sample-size 1000 --split random
python3 run.py experiment h3-fusion --sample-size 1000 --split random
python3 run.py experiment h4-compact-encoders --models distilbert,minilm --epochs 1
python3 run.py experiment h5-robustness --sample-size 1000
python3 run.py experiment h6-base-rates --sample-size 1000 --split random

# Orchestrated suites
python3 run.py experiment paper-v1 --profile smoke      # Fast validation (H1,H2,H3,H5,H6)
python3 run.py experiment paper-v1 --profile full       # Complete (includes H4 transformers)
python3 run.py experiment paper-v1 --profile confirmatory  # Final single-run test
```

## Hardware Requirements

### Minimum (CPU-only)
- Works for all experiments except H4 with reasonable times
- H4 transformer training will be extremely slow without GPU
- Recommended for smoke tests and baseline validation

### Recommended (GPU available)
- GPU strongly recommended for H4 compact encoder experiments
- Significantly reduces transformer training time (BERT, DistilBERT, MiniLM)
- Enables full experimental suite in reasonable timeframe

### Server Configuration
See `docs/SERVER_SETUP.md` for remote server setup instructions.

## Common Pitfalls and Solutions

### Issue: Kaggle authentication failures
**Solution**: Ensure kaggle.json is in one of:
- `~/.kaggle/kaggle.json`
- `~/.config/kaggle/kaggle.json`  
- `$KAGGLE_CONFIG_DIR/kaggle.json`

### Issue: Out of memory during transformer training
**Solution**: Reduce batch size in H4 experiments: `--batch-size 4` or `--batch-size 2`

### Issue: MeAJOR download failures
**Solution**: Check internet connection and Zenodo availability, use `--force-download` flag to retry

### Issue: Unrealistically high accuracy (100%)
**Solution**: STOP and audit data quality. Check for template leakage or synthetic data. Review `.artifacts/data/`

### Issue: Split policy confusion
**Solution**: 
- `--split random`: Standard random train/test split (may inflate metrics)
- `--split source`: Source-held-out evaluation (tests generalization)
- `--split leave-one-source-out`: Cross-source validation

## Paper Contribution Scope

### What's IN v1 (Core Contribution)
- Baseline reproduction (H0)
- Multi-source evaluation on MeAJOR (H1)
- URL/metadata feature ablation (H2)
- Calibrated fusion (H3)
- Compact encoders (H4)
- Robustness testing (H5)
- Realistic base rate simulation (H6)

### What's OUT of v1 (Future Work)
- RAG/local context integration
- LLM fallback in production
- QR code and attachment analysis
- PETA/user studies
- PT-BR as main contribution
- LLM->SLM distillation as primary focus

These are valid extensions but significantly increase scope. They should appear as "Future Work" not v1 requirements.

## Research Integrity Reminders

1. **Never report only the best result** from multiple runs - use confidence intervals
2. **Don't optimize for accuracy only** on balanced datasets - use operational metrics
3. **Don't claim deployment readiness** based only on random split performance
4. **Always separate exploratory and confirmatory phases** - don't cherry-pick
5. **Report limitations honestly** - especially Dandelion-NB reproduction failure
6. **Pre-seed all experiments** and report the seeds used
7. **Archive experimental data** for reproducibility (timestamped run folders)

## Emergency Procedures

If something goes wrong during execution:

1. **Stop immediately** if accuracy is exactly 100% - audit data quality
2. **Check logs** in `.artifacts/logs/` for detailed error information  
3. **Verify dataset integrity** by checking `.artifacts/data/` contents
4. **Validate environment** with `python3 run.py check` and `python3 run.py test`
5. **Backup results** before making major changes - copy timestamped run folders
6. **Document anomalies** in experimental notes before proceeding

---

**Last Updated**: 2025-06-03  
**Project Phase**: Experimental Implementation Complete, Awaiting Initial Results  
**Critical Path**: Kaggle Setup → Baseline Execution → Smoke Test → Exploratory Analysis → Confirmatory Test