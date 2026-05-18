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
.venv/bin/python run.py check
.venv/bin/python run.py test
```

## Run

Run the full paper reproduction pipeline:

```bash
.venv/bin/python run.py
```

Equivalent explicit command:

```bash
.venv/bin/python run.py train
```

The training step can be slow because it fine-tunes BERT and DistilBERT. A GPU
is strongly recommended for results close to the paper.

Run metadata is saved under `.artifacts/results/`, including dataset
provenance, class counts, split policy, model parameters and hardware details.
Each completed run is also saved under
`.artifacts/results/runs/baseline_paper/<timestamp>/`.

## Research Experiments

Run the first exploratory experiment on the MeAJOR multi-source dataset:

```bash
.venv/bin/python run.py experiment h1-meajor --sample-size 1000 --models nb,logreg --split random
```

Use source-held-out evaluation to test cross-source generalization:

```bash
.venv/bin/python run.py experiment h1-meajor --models nb,logreg,svm --split source
```

The experiment downloads MeAJOR v2.0 from Zenodo on first run and caches it
under `.artifacts/data/meajor/`. Metrics and metadata are written to
`.artifacts/results/research/` and timestamped run folders under
`.artifacts/results/runs/`.

The aggregate metrics table is:

```text
.artifacts/results/research/all_experiments_metrics.csv
```

Detailed execution protocol: `docs/EXPERIMENTS.md`.

Operational runbook for the paper workflow:
`docs/INSTRUCOES_EXECUCAO_PAPER.md`.

Run the implemented paper-v1 smoke suite:

```bash
.venv/bin/python run.py experiment paper-v1 --profile smoke
```

Run the heavier suite, including compact encoders:

```bash
.venv/bin/python run.py experiment paper-v1 --profile full
```

## Data Quality

Raw datasets are audited before feature extraction. Critical-risk datasets,
including template-based synthetic data, are rejected to avoid artificial
accuracy.

Manual quality check:

```bash
.venv/bin/python run.py test
```

## References

- Data setup: `docs/DATA_SETUP_GUIDE.md`
- Audit notes: `docs/README_AUDIT.md`
- Server setup: `docs/SERVER_SETUP.md`
- Source paper: `docs/Artigo_Phishing.pdf`
