# Phishing Email Detection

Python package for phishing email detection. The maintained default workflow is
scikit-learn first; BERT/DistilBERT remains optional.

## Layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-transformers.txt
├── docs/
├── notebooks/
├── scripts/
├── src/phishing_detection/
├── tests/
└── .artifacts/              # generated locally, ignored by git
```

Generated data, trained models, plots, logs, results and temp files are written
under `.artifacts/` instead of polluting the repository root.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Configure Kaggle credentials:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Validate the project:

```bash
phishing-check-setup
phishing-test-data-quality
```

## Run

Recommended sklearn pipeline:

```bash
phishing-train-sklearn
```

Optional full transformer pipeline:

```bash
pip install -r requirements-transformers.txt
phishing-train-full
```

Interactive helper:

```bash
scripts/setup_and_run.sh
```

## Data Quality

Raw datasets are audited before feature extraction. Critical-risk datasets,
including template-based synthetic data, are rejected to avoid artificial
accuracy.

Manual quality check:

```bash
phishing-test-data-quality
```

## References

- Data setup: `docs/DATA_SETUP_GUIDE.md`
- Audit notes: `docs/README_AUDIT.md`
- Server setup: `docs/SERVER_SETUP.md`
- Source paper: `docs/Artigo_Phishing.pdf`
