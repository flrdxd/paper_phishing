# Real Data Setup Guide

This project aims to reproduce the model comparison from `Artigo_Phishing.pdf`.
The full training flow includes Naive Bayes, Dandelion-optimized Naive Bayes,
BERT and DistilBERT.

Synthetic or template-based datasets are rejected before training because they
can produce artificial 100% accuracy.

The default Kaggle source is:

```text
naserabdullahalam/phishing-email-dataset
```

The pipeline expects a class distribution close to the paper's reported 42,891
phishing emails and 39,595 legitimate emails.

## 1. Configure Kaggle

Create a Kaggle account, generate an API token at
https://www.kaggle.com/settings, then install it locally:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Validate the API and project setup:

```bash
python3 run.py check
```

## 2. Install the Project

Use `uv` with the single requirements file:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## 3. Validate Data Quality

Run the data quality test suite before training:

```bash
python3 run.py test
```

Critical-risk datasets must not be used. Delete cached data and download again
if the audit reports synthetic data:

```bash
rm -f .artifacts/data/paper_phishing_dataset.csv
```

## 4. Train

Run the full paper reproduction pipeline:

```bash
python3 run.py train
```

This trains all comparison models from the paper:

- Naive Bayes
- Naive Bayes + Dandelion optimization
- BERT
- DistilBERT

## Expected Real-Data Signals

- Accuracy should be realistic, not automatically 100%.
- Training should take meaningful time, especially for BERT and DistilBERT.
- Confusion matrices should contain some false positives or false negatives.
- Cross-validation should show real variation.
- Important features should be contextually meaningful.

Generated data, models, plots, logs and results are stored in `.artifacts/`.
Dataset and run metadata are written to `.artifacts/results/`.
