# 🔐 Phishing Email Detection

**Optimized Machine Learning and Transformer Models for Phishing Detection**

Implementation based on the paper: *"Optimizing Phishing Detection: Comparative Analysis of Lightweight Machine Learning and Transformer Models"* (IEEE World Forum on Public Safety Technology 2025)

---

## 🎯 Overview

This project implements and optimizes multiple models for phishing email detection:

- **Naive Bayes** - Fast, lightweight baseline
- **Dandelion-Optimized Naive Bayes** - Nature-inspired hyperparameter optimization
- **BERT** - State-of-the-art transformer model
- **DistilBERT** - Optimized transformer for faster inference

**⚠️ IMPORTANT:** This system includes robust data quality checks to prevent artificial accuracy from synthetic datasets.

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
./setup_and_run.sh
```

This script automatically:
- ✅ Verifies environment setup
- ✅ Installs dependencies
- ✅ Checks Kaggle API configuration
- ✅ Downloads real datasets
- ✅ Runs the complete pipeline

### Option 2: Manual Setup

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Kaggle API
#    - Create account: https://www.kaggle.com/
#    - Get token: https://www.kaggle.com/settings
#    - Install token:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Verify setup
python check_kaggle_setup.py

# 5. Run pipeline
python src/main_sklearn_only.py
```

---

## 📊 Expected Results

With real datasets (Kaggle):

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Naive Bayes | 85-95% | 88-96% | 82-94% | 85-95% | 10-30s |
| NB + Dandelion | 88-97% | 90-98% | 86-96% | 88-97% | 30-120s |
| BERT | 95-99% | 96-99% | 94-99% | 95-99% | 30-60 min |
| DistilBERT | 94-99% | 95-99% | 93-99% | 94-99% | 20-45 min |

**⚠️ Note:** Results with 100% accuracy in <2 seconds indicate synthetic data and are automatically rejected.

---

## 🏗️ Project Structure

```
.
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── path_config.py           # Path configuration
│   ├── main.py                  # Complete pipeline
│   ├── main_sklearn_only.py     # Scikit-learn only pipeline
│   ├── models/
│   │   ├── naive_bayes.py       # Naive Bayes implementation
│   │   ├── dandelion_nb.py      # Dandelion-optimized NB
│   │   ├── bert_model.py        # BERT implementation
│   │   └── distilbert_model.py  # DistilBERT implementation
│   └── utils/
│       ├── data_auditor.py      # Data quality checks
│       ├── evaluation.py        # Model evaluation metrics
│       └── visualization.py     # Plotting and visualization
├── data/                        # Dataset storage
├── models/                      # Trained models
├── plots/                       # Generated visualizations
├── results/                     # Experiment results
├── logs/                        # Training logs
├── setup_and_run.sh            # Automated setup script
├── check_kaggle_setup.py       # Setup verification
└── test_data_quality.py        # Data quality tests
```

---

## 🛡️ Data Quality System

This project includes **automatic detection and prevention of synthetic datasets** that cause artificial accuracy:

### Quality Checks
- ✅ Text diversity validation (>30% required)
- ✅ Template repetition detection (<50% required)
- ✅ Vocabulary size verification (>100 words required)
- ✅ Duplicate detection
- ✅ Label distribution validation

### Automatic Protections
- ❌ Rejects synthetic datasets automatically
- ❌ Prevents fallback to synthetic data
- ❌ Alerts on suspicious training times
- ❌ Warns about perfect accuracy

**📖 See [README_AUDIT.md](README_AUDIT.md) for detailed information about the audit system.**

---

## 🔧 Usage

### Run Complete Pipeline (All Models)

```bash
source venv/bin/activate
python src/main.py
```

### Run Scikit-Learn Only (Faster, No GPU Required)

```bash
source venv/bin/activate
python src/main_sklearn_only.py
```

### Run Data Quality Checks

```bash
python test_data_quality.py
```

### Verify Kaggle Setup

```bash
python check_kaggle_setup.py
```

---

## 📈 Features

### Machine Learning Models
- **Optimized TF-IDF** vectorization with configurable parameters
- **Hyperparameter tuning** using grid search and random search
- **Cross-validation** with confidence intervals
- **Model calibration** for better probability estimates

### Transformer Models
- **Mixed precision training** for memory efficiency
- **Gradient accumulation** for larger effective batch sizes
- **GPU acceleration** support (CUDA)
- **Transfer learning** from pre-trained models

### Data Preprocessing
- **Text cleaning**: HTML removal, special character handling
- **URL/email/phone extraction** for feature engineering
- **Tokenization and stopword removal**
- **Data augmentation** (optional)

### Evaluation
- **Comprehensive metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Confusion matrices** and classification reports
- **Security metrics**: FPR, FNR, TPR, TNR
- **Visualizations**: ROC curves, learning curves, comparison charts

---

## 📊 Datasets

### Required Datasets
1. **Phishing Emails Dataset** - Kaggle: `subhajournal/phishingemails`
2. **SMS Spam Collection** - Kaggle: `uciml/sms-spam-collection-dataset`

### Data Quality
- ✅ Only real datasets from Kaggle are used
- ✅ Automatic quality validation
- ✅ Synthetic data rejection
- ✅ Proper train/test split with stratification

---

## 🐛 Troubleshooting

### "Kaggle API not configured"
```bash
# Follow these steps:
# 1. Create account at https://www.kaggle.com/
# 2. Get API token at https://www.kaggle.com/settings
# 3. Install token:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### "Synthetic dataset detected"
```bash
# Delete cached data and re-download:
rm -f data/phishing_emails.csv data/legitimate_emails.csv
python src/data_preprocessing.py
```

### "Module not found"
```bash
source venv/bin/activate
pip install pandas numpy scikit-learn nltk beautifulsoup4 kaggle kagglehub
```

### GPU/Out of Memory Errors
```bash
# Use scikit-learn only version (no GPU required):
python src/main_sklearn_only.py
```

---

## 📚 Documentation

- **[README_AUDIT.md](README_AUDIT.md)** - Quick guide to the audit system
- **[DATA_SETUP_GUIDE.md](DATA_SETUP_GUIDE.md)** - Detailed Kaggle setup guide
- **[AUDIT_SUMMARY.md](AUDIT_SUMMARY.md)** - Executive summary of technical audit
- **[AUDIT_REPORT_FINAL.md](AUDIT_REPORT_FINAL.md)** - Complete technical audit report
- **[STATUS.txt](STATUS.txt)** - Current system status

---

## 🎯 Research Context

This implementation reproduces and extends research from:

**Paper:** "Optimizing Phishing Detection: Comparative Analysis of Lightweight Machine Learning and Transformer Models"

**Conference:** IEEE World Forum on Public Safety Technology (WF-PST) 2025

**Contributions:**
- Comparative analysis of lightweight ML vs transformer models
- Optimization techniques for resource-constrained environments
- Comprehensive evaluation metrics for security applications
- Production-ready implementation with real-world deployment considerations

---

## ⚙️ Requirements

### Python Packages
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
beautifulsoup4>=4.9.0
kaggle>=1.5.0
kagglehub>=0.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
```

### Optional (for Transformers)
```
torch>=1.9.0
transformers>=4.0.0
```

### System Requirements
- **CPU:** Any modern processor
- **RAM:** 8GB+ recommended
- **GPU:** NVIDIA GPU with CUDA (for transformer models)
- **Storage:** 5GB+ free space

---

## 📝 License

This project is provided for research and educational purposes.

---

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. All code passes data quality checks
2. Real datasets are used (no synthetic data)
3. Results are properly validated
4. Documentation is updated

---

## 📧 Contact

For questions about the implementation or research, please refer to the original paper or open an issue in this repository.

---

**⚠️ IMPORTANT:** This system includes robust data quality validation. Results with 100% accuracy or training times <2 seconds are automatically flagged as potentially invalid. Always verify data quality before interpreting results.

**Last Updated:** 2026-05-12
**Status:** ✅ Production-Ready with Data Quality Safeguards