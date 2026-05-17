# Server Setup

## 1. Install Dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## 2. Configure Kaggle

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

Kaggle tokens can be generated at https://www.kaggle.com/settings.

## 3. Verify Setup

```bash
python3 run.py check
python3 run.py test
```

The checker verifies access to `naserabdullahalam/phishing-email-dataset`, the
Kaggle dataset used as the default source for the paper reproduction.

## 4. Train

Run the full paper reproduction pipeline:

```bash
python3 run.py train
```

This trains Naive Bayes, Dandelion-optimized Naive Bayes, BERT and DistilBERT.
A GPU is strongly recommended for the Transformer models.

## Notes

- Keep `.venv/` local to the server.
- Generated outputs are written to `.artifacts/`.
- Dataset and run metadata are written to `.artifacts/results/`.
- The Kaggle package is pinned in `requirements.txt` for authentication
  compatibility.
