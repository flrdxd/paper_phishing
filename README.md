# Phishing Email Detection

Research project for reproducing the model comparison from
`docs/Artigo_Phishing.pdf`.

The training flow compares all models discussed in the paper:

- Naive Bayes
- Naive Bayes + Dandelion optimization
- BERT
- DistilBERT

The default dataset is the Kaggle dataset referenced by the paper:
`naserabdullahalam/phishing-email-dataset`. The pipeline validates that the
class distribution is close to the paper's reported 42,891 phishing and 39,595
legitimate emails before training.

## Layout

```text
.
├── README.md
├── run.py
├── pyproject.toml
├── requirements.txt
├── docs/
├── notebooks/
├── src/phishing_detection/
├── tests/
└── .artifacts/              # generated locally, ignored by git
```

Generated data, trained models, plots, logs, results and temp files are written
under `.artifacts/` instead of polluting the repository root.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Configure Kaggle credentials:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Validate the project:

```bash
python3 run.py check
python3 run.py test
```

## Run

Run the full paper reproduction pipeline:

```bash
python3 run.py
```

Equivalent explicit command:

```bash
python3 run.py train
```

The training step can be slow because it fine-tunes BERT and DistilBERT. A GPU
is strongly recommended for results close to the paper.

Run metadata is saved under `.artifacts/results/`, including dataset
provenance, class counts, split policy, model parameters and hardware details.

## Data Quality

Raw datasets are audited before feature extraction. Critical-risk datasets,
including template-based synthetic data, are rejected to avoid artificial
accuracy.

Manual quality check:

```bash
python3 run.py test
```

## References

- Data setup: `docs/DATA_SETUP_GUIDE.md`
- Audit notes: `docs/README_AUDIT.md`
- Server setup: `docs/SERVER_SETUP.md`
- Source paper: `docs/Artigo_Phishing.pdf`
