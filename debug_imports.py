#!/usr/bin/env python3
"""
Debug script to test imports step by step and identify where the crash occurs
"""

import sys
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_import(module_name, import_statement):
    """Test a single import with detailed error handling"""
    logger.info(f"Testing import: {module_name}")
    try:
        exec(import_statement)
        logger.info(f"✓ {module_name} imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ {module_name} failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    logger.info("="*60)
    logger.info("DEBUGGING IMPORTS STEP BY STEP")
    logger.info("="*60)

    # Test basic imports
    imports = [
        ("Standard Library", [
            ("os", "import os"),
            ("sys", "import sys"),
            ("time", "import time"),
            ("re", "import re"),
        ]),
        ("Scientific Computing", [
            ("numpy", "import numpy as np"),
            ("pandas", "import pandas as pd"),
            ("scipy", "import scipy"),
        ]),
        ("Machine Learning", [
            ("sklearn", "import sklearn"),
            ("sklearn.feature_extraction", "from sklearn.feature_extraction.text import TfidfVectorizer"),
            ("sklearn.model_selection", "from sklearn.model_selection import train_test_split"),
            ("joblib", "import joblib"),
        ]),
        ("NLP", [
            ("nltk", "import nltk"),
            ("nltk.corpus", "from nltk.corpus import stopwords"),
            ("nltk.tokenize", "from nltk.tokenize import word_tokenize"),
        ]),
        ("Deep Learning", [
            ("torch", "import torch"),
            ("transformers", "import transformers"),
            ("accelerate", "import accelerate"),
        ]),
        ("Data Processing", [
            ("bs4", "from bs4 import BeautifulSoup"),
            ("lxml", "import lxml"),
        ]),
        ("Visualization", [
            ("matplotlib", "import matplotlib.pyplot as plt"),
            ("seaborn", "import seaborn as sns"),
            ("wordcloud", "from wordcloud import WordCloud"),
        ]),
        ("Data Download", [
            ("kagglehub", "import kagglehub"),
        ]),
        ("Project Imports", [
            ("data_preprocessing", "from src.data_preprocessing import DataPreprocessor"),
            ("naive_bayes", "from src.models.naive_bayes import NaiveBayesPhishingDetector"),
            ("dandelion_nb", "from src.models.dandelion_nb import DandelionNaiveBayesDetector"),
        ]),
    ]

    failed_imports = []

    for category, category_imports in imports:
        logger.info(f"\n{'='*40}")
        logger.info(f"Category: {category}")
        logger.info(f"{'='*40}")

        for module_name, import_statement in category_imports:
            success = test_import(module_name, import_statement)
            if not success:
                failed_imports.append(module_name)

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    if failed_imports:
        logger.error(f"Failed imports: {', '.join(failed_imports)}")
    else:
        logger.info("✓ All imports successful!")

    return len(failed_imports) == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)