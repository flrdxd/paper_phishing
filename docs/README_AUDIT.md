# Data Audit Notes

The project includes safeguards against synthetic or template-based phishing
datasets. This matters because trivial synthetic data can produce invalid
research results, including artificial 100% accuracy.

The standard dataset is `naserabdullahalam/phishing-email-dataset`, matching
the dataset reference used by the paper. The pipeline rejects datasets whose
class counts are too far from the paper's reported distribution.

## Active Protections

- Low text diversity is flagged.
- High template repetition is flagged.
- Very small vocabulary is flagged.
- Critical-risk datasets are rejected before preprocessing.
- Data quality tests can be run with `python3 run.py test`.

## Standard Workflow

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
python3 run.py check
python3 run.py test
python3 run.py train
```

The training command runs all paper comparison models: Naive Bayes, Dandelion,
BERT and DistilBERT.

## Warning Signs

- Training finishes unrealistically fast for the full model set.
- Accuracy is exactly 100%.
- The confusion matrix has no errors.
- Cross-validation has no meaningful variation.
- Top features are obvious template tokens rather than useful language signals.

If these appear, stop and inspect the cached data in `.artifacts/data/`.
