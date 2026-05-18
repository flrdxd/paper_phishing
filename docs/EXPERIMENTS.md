# Experiment Protocol

This document defines the execution order, commands, outputs, and decision
criteria for the phishing detection research track.

## 1. Validate Environment

```bash
source .venv/bin/activate
.venv/bin/python run.py test
.venv/bin/python run.py check
```

`run.py check` must pass before running the paper baseline because the baseline
uses the Kaggle dataset. MeAJOR experiments do not require Kaggle credentials.

Accepted Kaggle credential locations:

```text
~/.kaggle/kaggle.json
~/.config/kaggle/kaggle.json
$KAGGLE_CONFIG_DIR/kaggle.json
```

## 2. H0 - Paper Baseline Reproduction

Command:

```bash
.venv/bin/python run.py train
```

Purpose:

- reproduce Naive Bayes, Dandelion-NB, BERT, and DistilBERT;
- compare against `docs/Artigo_Phishing.pdf`;
- freeze the baseline before new hypotheses.

Outputs:

```text
.artifacts/results/model_results.csv
.artifacts/results/model_results.json
.artifacts/results/results_summary.txt
.artifacts/results/run_metadata.json
.artifacts/results/runs/baseline_paper/<timestamp>/
```

Primary interpretation:

- BERT and DistilBERT should be close to the paper.
- Dandelion-NB is expected to be reported honestly if it does not reproduce the
  paper's large gain.

## 3. H1 - MeAJOR Generalization

Fast smoke test:

```bash
.venv/bin/python run.py experiment h1-meajor --sample-size 1000 --models nb,logreg --split random
```

Source-held-out smoke test:

```bash
.venv/bin/python run.py experiment h1-meajor --sample-size 1000 --models nb --split source
```

Full H1 runs:

```bash
.venv/bin/python run.py experiment h1-meajor --models nb,logreg,svm --split random
.venv/bin/python run.py experiment h1-meajor --models nb,logreg,svm --split source
```

Outputs:

```text
.artifacts/results/research/h1_meajor_metrics.csv
.artifacts/results/research/h1_meajor_metrics.json
.artifacts/results/research/h1_meajor_metadata.json
.artifacts/results/runs/h1_meajor/<timestamp>/
```

Decision criteria:

- If source split drops F1 or raises FPR substantially, generalization becomes a
  central argument for the paper.
- If random split remains high but source split degrades, avoid claiming the
  model is deployment-ready based only on random split.

## 4. Metrics Required for Every Experiment

Every run should save:

- Accuracy;
- Precision;
- Recall;
- F1;
- MCC;
- AUC-PR when scores are available;
- ROC-AUC when available;
- TN, FP, FN, TP;
- FPR, FNR, TNR;
- `FPR@Recall>=98%` when scores are available;
- training time;
- inference time;
- dataset, split, sample count, seed, command, git commit, hardware.

Aggregate table:

```text
.artifacts/results/research/all_experiments_metrics.csv
```

## 5. Implemented Paper-v1 Experiments

Individual commands:

```bash
.venv/bin/python run.py experiment h2-url-meta --sample-size 1000 --split random
.venv/bin/python run.py experiment h3-fusion --sample-size 1000 --split random
.venv/bin/python run.py experiment h4-compact-encoders --sample-size 1000 --models distilbert,minilm --epochs 1
.venv/bin/python run.py experiment h5-robustness --sample-size 1000
.venv/bin/python run.py experiment h6-base-rates --sample-size 1000 --split random
```

Suite commands:

```bash
.venv/bin/python run.py experiment paper-v1 --profile smoke
.venv/bin/python run.py experiment paper-v1 --profile full
```

`smoke` runs H1, H2, H3, H5, and H6 on 1,000 samples. H4 compact encoders are
excluded from smoke because they download and fine-tune transformer models.
`full` includes H4 with DistilBERT and MiniLM.

Before writing final analysis, run H0 baseline and `paper-v1 --profile full`.
