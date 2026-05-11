"""
Safe Main Execution Script for Phishing Email Detection
This version includes extensive logging and error handling for debugging
"""

import os
import sys
import time
import traceback
import logging

# Import path configuration first
from path_config import PATHS

# Set up logging BEFORE any other imports
log_file = os.path.join(PATHS['LOGS_DIR'], 'phishing_detection_safe.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def flush_log():
    """Force flush the log file"""
    for handler in logger.handlers:
        handler.flush()

def safe_import(module_name, import_func):
    """Safely import a module with detailed error handling"""
    logger.info(f"Attempting to import: {module_name}")
    try:
        result = import_func()
        logger.info(f"✓ Successfully imported: {module_name}")
        flush_log()
        return result
    except Exception as e:
        logger.error(f"✗ Failed to import {module_name}: {type(e).__name__}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flush_log()
        raise

# Import with error handling
logger.info("="*60)
logger.info("STARTING SAFE IMPORT PHASE")
logger.info("="*60)
flush_log()

# Standard library
safe_import("os", lambda: __import__('os'))
safe_import("sys", lambda: __import__('sys'))
safe_import("time", lambda: __import__('time'))
flush_log()

# Scientific computing
safe_import("numpy", lambda: __import__('numpy'))
safe_import("pandas", lambda: __import__('pandas'))
flush_log()

# Machine Learning
safe_import("sklearn", lambda: __import__('sklearn'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
logger.info("✓ sklearn components imported")
flush_log()

# NLP
safe_import("nltk", lambda: __import__('nltk'))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
logger.info("✓ nltk components imported")
flush_log()

# Deep Learning (most likely to cause bus errors)
logger.info("Attempting to import deep learning libraries...")
try:
    import torch
    logger.info(f"✓ PyTorch imported successfully. Version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    logger.error(f"✗ PyTorch import failed: {e}")
    traceback.print_exc()
flush_log()

try:
    import transformers
    logger.info(f"✓ Transformers imported successfully. Version: {transformers.__version__}")
except Exception as e:
    logger.error(f"✗ Transformers import failed: {e}")
    traceback.print_exc()
flush_log()

try:
    import accelerate
    logger.info(f"✓ Accelerate imported successfully")
except Exception as e:
    logger.error(f"✗ Accelerate import failed: {e}")
    traceback.print_exc()
flush_log()

# Data processing
safe_import("bs4", lambda: __import__('bs4'))
safe_import("lxml", lambda: __import__('lxml'))
from bs4 import BeautifulSoup
logger.info("✓ bs4 components imported")
flush_log()

# Data download
try:
    import kagglehub
    logger.info("✓ KaggleHub imported successfully")
except Exception as e:
    logger.error(f"✗ KaggleHub import failed: {e}")
    traceback.print_exc()
flush_log()

# Visualization
safe_import("matplotlib.pyplot", lambda: __import__('matplotlib.pyplot'))
safe_import("seaborn", lambda: __import__('seaborn'))
try:
    from wordcloud import WordCloud
    logger.info("✓ WordCloud imported successfully")
except Exception as e:
    logger.error(f"✗ WordCloud import failed: {e}")
    traceback.print_exc()
flush_log()

# Project imports
logger.info("Attempting to import project modules...")
try:
    from data_preprocessing import DataPreprocessor
    logger.info("✓ DataPreprocessor imported")
except Exception as e:
    logger.error(f"✗ DataPreprocessor import failed: {e}")
    traceback.print_exc()
flush_log()

try:
    from models.naive_bayes import NaiveBayesPhishingDetector
    logger.info("✓ NaiveBayesPhishingDetector imported")
except Exception as e:
    logger.error(f"✗ NaiveBayesPhishingDetector import failed: {e}")
    traceback.print_exc()
flush_log()

try:
    from models.dandelion_nb import DandelionNaiveBayesDetector
    logger.info("✓ DandelionNaiveBayesDetector imported")
except Exception as e:
    logger.error(f"✗ DandelionNaiveBayesDetector import failed: {e}")
    traceback.print_exc()
flush_log()

logger.info("="*60)
logger.info("ALL IMPORTS COMPLETED SUCCESSFULLY")
logger.info("="*60)
flush_log()

# Now try the actual code
class PhishingDetectionPipelineSafe:
    """Safe version with extensive error handling"""

    def __init__(self, max_features=5000, random_state=42):
        logger.info("Initializing pipeline...")
        self.max_features = max_features
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.results = {}
        flush_log()

    def load_and_preprocess_data(self, force_download=False):
        """Safe data loading with detailed logging"""
        logger.info("Starting data preprocessing...")
        flush_log()

        try:
            self.preprocessor = DataPreprocessor(
                max_features=self.max_features,
                random_state=self.random_state
            )
            logger.info("DataPreprocessor created")
            flush_log()

            logger.info("Attempting to download dataset...")
            flush_log()
            df = self.preprocessor.download_dataset(force_download=force_download)
            logger.info(f"Dataset downloaded. Shape: {df.shape}")
            flush_log()

            logger.info("Preprocessing dataframe...")
            flush_log()
            df = self.preprocessor.preprocess_dataframe(df)
            logger.info(f"Dataframe preprocessed. Shape: {df.shape}")
            flush_log()

            logger.info("Splitting data...")
            flush_log()
            X_train_ml, X_test_ml, y_train_ml, y_test_ml = self.preprocessor.split_for_ml(df, test_size=0.3)
            logger.info("ML data split complete")
            flush_log()

            X_train_tf, X_test_tf, y_train_tf, y_test_tf = self.preprocessor.split_for_transformer(df, test_size=0.2)
            logger.info("Transformer data split complete")
            flush_log()

            return (X_train_ml, X_test_ml, y_train_ml, y_test_ml,
                    X_train_tf, X_test_tf, y_train_tf, y_test_tf)

        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            traceback.print_exc()
            flush_log()
            raise

    def train_naive_bayes_model(self, X_train, X_test, y_train, y_test):
        """Train Naive Bayes with error handling"""
        logger.info("Training Naive Bayes model...")
        flush_log()

        try:
            nb_detector = NaiveBayesPhishingDetector(
                alpha=1.0,
                fit_prior=True,
                random_state=self.random_state
            )
            logger.info("NaiveBayesPhishingDetector created")
            flush_log()

            logger.info("Starting training...")
            flush_log()
            nb_detector.train(X_train, y_train)
            logger.info("Training complete")
            flush_log()

            logger.info("Evaluating model...")
            flush_log()
            metrics = nb_detector.evaluate(X_test, y_test)
            logger.info(f"Results: {metrics}")
            flush_log()

            self.models['naive_bayes'] = nb_detector
            return metrics

        except Exception as e:
            logger.error(f"Naive Bayes training failed: {e}")
            traceback.print_exc()
            flush_log()
            raise

def main():
    """Main function with comprehensive error handling"""
    logger.info("="*60)
    logger.info("STARTING SAFE PHISHING DETECTION PIPELINE")
    logger.info("="*60)
    flush_log()

    try:
        # Initialize pipeline
        logger.info("Creating pipeline instance...")
        flush_log()
        pipeline = PhishingDetectionPipelineSafe(max_features=5000, random_state=42)
        logger.info("Pipeline created successfully")
        flush_log()

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        flush_log()
        (X_train_ml, X_test_ml, y_train_ml, y_test_ml,
         X_train_tf, X_test_tf, y_train_tf, y_test_tf) = pipeline.load_and_preprocess_data()
        logger.info("Data preprocessing complete")
        flush_log()

        # Train Naive Bayes
        logger.info("Training Naive Bayes...")
        flush_log()
        results = pipeline.train_naive_bayes_model(
            X_train_ml, X_test_ml, y_train_ml, y_test_ml
        )
        logger.info("Naive Bayes training complete")
        flush_log()

        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Results: {results}")
        flush_log()

        return results

    except Exception as e:
        logger.critical(f"CRITICAL ERROR IN MAIN: {type(e).__name__}: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        flush_log()
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)