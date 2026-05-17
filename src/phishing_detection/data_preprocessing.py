"""
Data Preprocessing Module for Phishing Email Detection - FIXED VERSION

This module handles:
- Dataset download from Kaggle with robust error handling
- Text cleaning (HTML removal, special characters)
- Tokenization and stopwords removal
- TF-IDF vectorization
- Train/test split for ML and transformer models

CRITICAL FIXES:
- Eliminated all data leakage sources
- Fixed all NaN/None handling issues
- Robust validation at every step
- No artificial accuracy possible
- Proper error handling and logging
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import kagglehub
import logging

# Import path configuration
from phishing_detection.path_config import PATHS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_duration(seconds):
    """Format seconds as HH:MM:SS for progress logs."""
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

PAPER_DATASET_ID = "naserabdullahalam/phishing-email-dataset"
PAPER_EXPECTED_PHISHING = 42891
PAPER_EXPECTED_LEGITIMATE = 39595
PAPER_EXPECTED_TOTAL = PAPER_EXPECTED_PHISHING + PAPER_EXPECTED_LEGITIMATE
PAPER_COUNT_TOLERANCE = 0.20

# Download NLTK data
logger.info("Downloading required NLTK data...")

# Download punkt (older versions)
try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK punkt already downloaded")
except LookupError:
    logger.info("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=False)
    logger.info("NLTK punkt downloaded successfully")

# Download punkt_tab (newer versions)
try:
    nltk.data.find('tokenizers/punkt_tab')
    logger.info("NLTK punkt_tab already downloaded")
except LookupError:
    logger.info("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', quiet=False)
    logger.info("NLTK punkt_tab downloaded successfully")

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
    logger.info("NLTK stopwords already downloaded")
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=False)
    logger.info("NLTK stopwords downloaded successfully")


class DataPreprocessor:
    """Handles data preprocessing for phishing email detection."""

    def __init__(self, max_features=5000, random_state=42):
        """
        Initialize preprocessor with robust defaults.

        Args:
            max_features: Maximum number of TF-IDF features
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.stop_words = set(stopwords.words('english'))

        # Suspicious TLDs for phishing detection
        self.suspicious_tlds = {
            '.tk', '.top', '.xyz', '.info', '.biz', '.online',
            '.club', '.site', '.download', '.racing', '.zip',
            '.win', '.bet', '.game', '.accountant', '.faith',
            '.help', '.ltd', '.stream', '.mom', '.date',
            '.review', '.science', '.loan', '.credit', '.finance',
            '.money', '.cash', '.bank', '.fund', '.trade',
            '.crypto', '.invest', '.stock', '.forex', '.tech'
        }

    def _log_progress(self, phase, completed, total, start_time):
        """Log progress for long preprocessing steps."""
        if completed % 5000 != 0 and completed != total:
            return

        elapsed = time.time() - start_time
        progress = completed / total
        eta = (elapsed / progress) - elapsed if progress > 0 else 0

        logger.info(
            "%s progress: %s/%s rows (%.1f%%), elapsed=%s, eta=%s",
            phase,
            completed,
            total,
            progress * 100,
            format_duration(elapsed),
            format_duration(eta),
        )

    def _apply_text_step(self, texts, function, phase):
        """Apply text processing with periodic progress logs."""
        total = len(texts)
        start_time = time.time()
        processed = []

        for index, text in enumerate(texts, start=1):
            processed.append(function(text))
            self._log_progress(phase, index, total, start_time)

        logger.info("%s complete in %s", phase, format_duration(time.time() - start_time))
        return processed

    def _safe_csv_read(self, filepath, encoding='utf-8'):
        """
        Safely read CSV with comprehensive error handling.

        Args:
            filepath: Path to CSV file
            encoding: File encoding

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            logger.info(f"✓ Successfully loaded {filepath} with encoding {encoding}")
            return df
        except UnicodeDecodeError:
            logger.warning(f"⚠️  Failed to read {filepath} with {encoding}, trying latin-1")
            df = pd.read_csv(filepath, encoding='latin-1')
            logger.info(f"✓ Successfully loaded {filepath} with encoding latin-1")
            return df
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to read {filepath}: {e}")
            raise RuntimeError(f"Failed to read CSV file {filepath}: {e}")

    def _validate_dataframe_structure(self, df, name="dataset"):
        """
        CRITICAL: Validate dataframe structure before processing.

        Args:
            df: DataFrame to validate
            name: Name of dataset for logging

        Raises:
            ValueError: If dataframe structure is invalid
        """
        logger.info(f"\n" + "="*60)
        logger.info(f"VALIDATING {name.upper()} STRUCTURE")
        logger.info("="*60)

        # Check 1: Not empty
        if len(df) == 0:
            raise ValueError(f"{name} is empty (0 rows)")

        # Check 2: Has columns
        if len(df.columns) == 0:
            raise ValueError(f"{name} has no columns")

        logger.info(f"✓ {name} structure: {df.shape} with columns {df.columns.tolist()[:5]}...")

        # Check 3: No all-NaN columns
        all_nan_mask = df.isna().all()
        all_nan_cols = df.columns[all_nan_mask].tolist()
        if all_nan_cols:
            logger.error(f"❌ CRITICAL: {name} has completely NaN columns: {all_nan_cols}")
            raise ValueError(f"{name} has invalid columns with all NaN values")

        logger.info("✓ Dataset structure validation passed")
        logger.info("="*60 + "\n")

    def _clean_text_column(self, df, text_col='text'):
        """
        CRITICAL: Clean and validate text column.

        Args:
            df: DataFrame with text column
            text_col: Name of text column

        Returns:
            pd.DataFrame: DataFrame with cleaned text
        """
        logger.info(f"Cleaning text column '{text_col}'...")

        # Check column exists
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in dataframe")

        # Convert to string and handle NaN
        df[text_col] = df[text_col].astype(str).fillna('')

        # Remove completely empty strings
        empty_before = (df[text_col].str.len() == 0).sum()
        if empty_before > 0:
            logger.warning(f"⚠️  Found {empty_before} empty strings in {text_col}")
            df = df[df[text_col].str.len() > 0].reset_index(drop=True)

        # Log statistics
        avg_length = df[text_col].str.len().mean()
        max_length = df[text_col].str.len().max()
        min_length = df[text_col].str.len().min()

        logger.info(f"✓ Text statistics - Avg: {avg_length:.1f}, Max: {max_length}, Min: {min_length}")
        logger.info(f"✓ Cleaned {text_col} - removed {empty_before} empty strings")

        return df

    def _find_dataset_csv(self, dataset_dir):
        """Find the most likely CSV file in a downloaded Kaggle dataset."""
        csv_files = []
        for root, _, files in os.walk(dataset_dir):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, filename))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in downloaded dataset: {dataset_dir}")

        csv_files.sort(key=lambda path: os.path.getsize(path), reverse=True)
        return csv_files[0]

    def _select_text_column(self, df, label_column=None):
        """Select the most plausible email text column."""
        preferred_terms = ("email text", "text", "message", "body", "content", "email")
        for term in preferred_terms:
            for col in df.columns:
                if col == label_column:
                    continue
                if term in col.lower():
                    return col

        candidates = [col for col in df.columns if col != label_column]
        if not candidates:
            raise ValueError("Could not identify a text column in the paper dataset")

        return max(candidates, key=lambda col: df[col].astype(str).str.len().mean())

    def _select_label_column(self, df):
        """Select the most plausible phishing/legitimate label column."""
        preferred_terms = ("email type", "label", "class", "category", "type", "target")
        for term in preferred_terms:
            for col in df.columns:
                if term in col.lower():
                    return col

        for col in df.columns:
            values = df[col].dropna().astype(str).str.lower().head(500)
            matched = values.str.contains("phishing|spam|safe|legitimate|ham|malicious", regex=True).mean()
            if matched > 0.5:
                return col

        raise ValueError("Could not identify a label column in the paper dataset")

    def _map_labels_to_binary(self, series):
        """Map common phishing dataset labels to 1=phishing, 0=legitimate."""
        normalized = series.astype(str).str.lower().str.strip()

        def map_label(value):
            if value in {"1", "1.0"}:
                return 1
            if value in {"0", "0.0"}:
                return 0
            if any(token in value for token in ("phishing", "malicious", "spam")):
                return 1
            if any(token in value for token in ("safe", "legitimate", "ham", "not spam", "benign")):
                return 0
            return np.nan

        return normalized.apply(map_label)

    def _load_paper_dataset(self, dataset_dir):
        """Load and normalize the Kaggle dataset referenced by the paper."""
        dataset_path = self._find_dataset_csv(dataset_dir)
        raw_df = self._safe_csv_read(dataset_path)
        self._validate_dataframe_structure(raw_df, "paper dataset")

        label_col = self._select_label_column(raw_df)
        text_col = self._select_text_column(raw_df, label_col)
        logger.info(f"✓ Paper dataset columns - text: '{text_col}', label: '{label_col}'")

        df = raw_df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = self._map_labels_to_binary(df["label"])
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Paper dataset normalization produced no usable rows")

        return df, dataset_path

    def _validate_paper_distribution(self, df):
        """Reject datasets that are too far from the paper's reported counts."""
        counts = df["label"].value_counts().to_dict()
        phishing_count = counts.get(1, 0)
        legitimate_count = counts.get(0, 0)
        total = len(df)

        logger.info("Paper dataset target distribution:")
        logger.info(f"  Expected: {PAPER_EXPECTED_PHISHING} phishing, {PAPER_EXPECTED_LEGITIMATE} legitimate")
        logger.info(f"  Actual:   {phishing_count} phishing, {legitimate_count} legitimate")

        lower_total = PAPER_EXPECTED_TOTAL * (1 - PAPER_COUNT_TOLERANCE)
        upper_total = PAPER_EXPECTED_TOTAL * (1 + PAPER_COUNT_TOLERANCE)
        if total < lower_total or total > upper_total:
            raise ValueError(
                f"Dataset total ({total}) is too far from the paper total "
                f"({PAPER_EXPECTED_TOTAL}). Check the Kaggle source."
            )

        for name, actual, expected in (
            ("phishing", phishing_count, PAPER_EXPECTED_PHISHING),
            ("legitimate", legitimate_count, PAPER_EXPECTED_LEGITIMATE),
        ):
            lower = expected * (1 - PAPER_COUNT_TOLERANCE)
            upper = expected * (1 + PAPER_COUNT_TOLERANCE)
            if actual < lower or actual > upper:
                raise ValueError(
                    f"Dataset {name} count ({actual}) is too far from the paper count ({expected})."
                )

    def _write_dataset_metadata(self, df, source_path):
        """Save dataset provenance for reproducibility."""
        metadata = {
            "dataset_id": PAPER_DATASET_ID,
            "source_path": source_path,
            "expected_counts": {
                "phishing": PAPER_EXPECTED_PHISHING,
                "legitimate": PAPER_EXPECTED_LEGITIMATE,
                "total": PAPER_EXPECTED_TOTAL,
            },
            "actual_counts": {
                "phishing": int((df["label"] == 1).sum()),
                "legitimate": int((df["label"] == 0).sum()),
                "total": int(len(df)),
            },
            "random_state": self.random_state,
            "max_features": self.max_features,
            "label_mapping": {"phishing": 1, "legitimate": 0},
        }

        output_path = os.path.join(PATHS["RESULTS_DIR"], "dataset_metadata.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(metadata, file, indent=4)
        logger.info(f"✓ Dataset metadata saved to {output_path}")

    def download_dataset(self, force_download=False):
        """
        Download phishing email dataset from Kaggle with robust error handling.

        Returns:
            pd.DataFrame: Combined dataset with 'text' and 'label' columns
        """
        data_dir = PATHS['DATA_DIR']
        os.makedirs(data_dir, exist_ok=True)

        paper_file = os.path.join(data_dir, 'paper_phishing_dataset.csv')

        # Download or use cache
        if not force_download and os.path.exists(paper_file):
            logger.info("Dataset already exists. Skipping download.")
            logger.warning("⚠️  WARNING: Using cached dataset. Use force_download=True to re-download.")

            try:
                combined_df = self._safe_csv_read(paper_file)
                self._validate_dataframe_structure(combined_df, "cached paper dataset")
                self._validate_paper_distribution(combined_df)
            except Exception as e:
                logger.error(f"❌ CRITICAL: Failed to load cached dataset: {e}")
                logger.error("❌ Deleting corrupted cache files...")
                os.remove(paper_file)
                logger.info("✓ Cache files deleted, will download fresh data")
                force_download = True

        if force_download or not os.path.exists(paper_file):
            logger.info(f"Downloading paper dataset from Kaggle: {PAPER_DATASET_ID}")

            try:
                dataset_dir = kagglehub.dataset_download(PAPER_DATASET_ID)
                combined_df, source_path = self._load_paper_dataset(dataset_dir)
                self._validate_paper_distribution(combined_df)
                combined_df.to_csv(paper_file, index=False)
                self._write_dataset_metadata(combined_df, source_path)

                logger.info(f"✓ Dataset downloaded and cached to {data_dir}")

            except Exception as e:
                logger.error(f"❌ CRITICAL: Error downloading dataset: {e}")
                raise RuntimeError(f"Failed to download paper dataset: {e}")

        # CRITICAL: Final comprehensive validation before returning
        logger.info("\n" + "="*60)
        logger.info("FINAL DATA VALIDATION")
        logger.info("="*60)

        for col in ['text', 'label']:
            if col not in combined_df.columns:
                raise ValueError(f"Paper dataset missing required column '{col}'")

        combined_df['text'] = combined_df['text'].fillna('').astype(str)
        combined_df = combined_df[combined_df['text'].str.len() > 0].reset_index(drop=True)
        combined_df['label'] = combined_df['label'].astype(int)

        if not set(combined_df['label'].unique()).issubset({0, 1}):
            raise ValueError(f"Paper dataset has invalid labels: {combined_df['label'].unique()}")

        # Remove duplicates
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        duplicates_removed = initial_count - len(combined_df)

        if duplicates_removed > 0:
            logger.info(f"✓ Removed {duplicates_removed} duplicate samples")

        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Final statistics
        final_phishing = len(combined_df[combined_df['label'] == 1])
        final_legitimate = len(combined_df[combined_df['label'] == 0])

        logger.info(f"✓ Final combined dataset: {len(combined_df)} samples")
        logger.info(f"✓ Distribution: {final_phishing} phishing, {final_legitimate} legitimate")

        # Sanity check for realistic accuracy
        if final_phishing == 0 or final_legitimate == 0:
            raise ValueError("Dataset contains only one class - cannot train meaningful model")
        self._validate_paper_distribution(combined_df)
        self._write_dataset_metadata(combined_df, paper_file)

        logger.info("✓ All data validation checks passed")
        logger.info("="*60 + "\n")

        return combined_df

    def clean_text(self, text):
        """
        Clean text by removing HTML, special characters, etc.

        Args:
            text: Raw text string

        Returns:
            str: Cleaned text
        """
        # Handle NaN/None
        if pd.isna(text) or text is None:
            return ""

        # Convert to string
        text = str(text)

        # Remove HTML tags
        try:
            text = BeautifulSoup(text, 'lxml').get_text()
        except Exception as e:
            logger.warning(f"⚠️  HTML parsing failed: {e}, using original text")

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace again
        text = ' '.join(text.split())

        return text

    def tokenize_and_remove_stopwords(self, text):
        """
        Tokenize text and remove stopwords.

        Args:
            text: Cleaned text string

        Returns:
            str: Text with stopwords removed
        """
        # Handle empty text
        if not text or len(text.strip()) == 0:
            return ""

        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"⚠️  NLTK tokenization failed: {e}, using simple tokenization")
            tokens = text.split()

        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]

        return ' '.join(tokens)

    def preprocess_dataframe(self, df):
        """
        Preprocess entire DataFrame with comprehensive validation.

        Args:
            df: DataFrame with 'text' and 'label' columns

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING DATAFRAME")
        logger.info("="*60)

        # CRITICAL: Validate input dataframe
        self._validate_dataframe_structure(df, "input dataframe")

        # Check required columns
        for col in ['text', 'label']:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from input dataframe")

        # Log initial statistics
        logger.info(f"✓ Input dataset: {len(df)} samples")
        logger.info(f"✓ Label distribution: {df['label'].value_counts().to_dict()}")

        # Run quality audit before deriving features.
        from phishing_detection.utils.data_auditor import audit_dataframe

        audit_results = audit_dataframe(df, "Raw training dataset")
        if audit_results['risk_level'] == 'CRITICAL':
            issues = '; '.join(audit_results['critical_issues'])
            raise ValueError(f"Dataset failed quality audit: {issues}")
        if audit_results['risk_level'] in {'HIGH', 'MEDIUM'}:
            logger.warning(f"⚠️  Dataset audit risk level: {audit_results['risk_level']}")

        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text'] = self._apply_text_step(
            df['text'].tolist(),
            self.clean_text,
            "Text cleaning"
        )

        # Tokenize and remove stopwords
        logger.info("Tokenizing and removing stopwords...")
        df['processed_text'] = self._apply_text_step(
            df['cleaned_text'].tolist(),
            self.tokenize_and_remove_stopwords,
            "Tokenization"
        )

        # Remove empty processed texts
        initial_count = len(df)
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
        removed_count = initial_count - len(df)

        if removed_count > 0:
            logger.info(f"✓ Removed {removed_count} samples with empty processed text")

        # Final validation
        if len(df) == 0:
            raise ValueError("All samples were removed during preprocessing - dataset is invalid")

        logger.info(f"✓ Preprocessing complete - {len(df)} samples remaining")
        logger.info("="*60 + "\n")

        return df

    def tfidf_vectorize(self, texts, fit=True):
        """
        Convert texts to TF-IDF vectors.

        Args:
            texts: List of text strings
            fit: Whether to fit vectorizer (True for training data)

        Returns:
            np.ndarray: TF-IDF vectors
        """
        # Validate input
        if not texts or len(texts) == 0:
            raise ValueError("Cannot vectorize empty text list")

        if fit or self.tfidf_vectorizer is None:
            logger.info("Fitting TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=5,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"✓ TF-IDF fitted with shape: {tfidf_matrix.shape}")
        else:
            logger.info("Transforming texts with existing TF-IDF vectorizer...")
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            logger.info(f"✓ TF-IDF transformed with shape: {tfidf_matrix.shape}")

        return tfidf_matrix

    def split_for_ml(self, df, test_size=0.3):
        """
        Split data for ML models with stratification.

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test (TF-IDF vectors and labels)
        """
        logger.info(f"Splitting data for ML models (test_size={test_size})...")

        # Validate dataframe
        self._validate_dataframe_structure(df, "ML split dataframe")

        # Extract features and labels
        X = df['processed_text']
        y = df['label']

        # Validate we have both classes
        unique_labels = y.unique()
        if len(unique_labels) < 2:
            raise ValueError(f"Dataset only has {len(unique_labels)} class(es): {unique_labels}")

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # TF-IDF vectorization
        X_train_tfidf = self.tfidf_vectorize(X_train.tolist(), fit=True)
        X_test_tfidf = self.tfidf_vectorize(X_test.tolist(), fit=False)

        # Log split statistics
        logger.info(f"✓ Train samples: {X_train_tfidf.shape[0]}")
        logger.info(f"✓ Test samples: {X_test_tfidf.shape[0]}")
        logger.info(f"✓ Train labels: {y_train.value_counts().to_dict()}")
        logger.info(f"✓ Test labels: {y_test.value_counts().to_dict()}")

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def split_for_transformer(self, df, test_size=0.2):
        """
        Split data for transformer models.

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test (texts and labels)
        """
        logger.info(f"Splitting data for transformer models (test_size={test_size})...")

        # Validate dataframe
        self._validate_dataframe_structure(df, "transformer split dataframe")

        # Extract features and labels
        X = df['cleaned_text']  # Use cleaned text for transformers
        y = df['label']

        # Validate we have both classes
        unique_labels = y.unique()
        if len(unique_labels) < 2:
            raise ValueError(f"Dataset only has {len(unique_labels)} class(es): {unique_labels}")

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Log split statistics
        logger.info(f"✓ Train samples: {len(X_train)}")
        logger.info(f"✓ Test samples: {len(X_test)}")
        logger.info(f"✓ Train labels: {y_train.value_counts().to_dict()}")
        logger.info(f"✓ Test labels: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def get_texts_by_label(self, df):
        """
        Get texts separated by label for word cloud generation.

        Args:
            df: Preprocessed DataFrame

        Returns:
            tuple: phishing_texts, legitimate_texts
        """
        # Validate dataframe
        if 'processed_text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataframe missing required columns for text extraction")

        phishing_texts = df[df['label'] == 1]['processed_text'].tolist()
        legitimate_texts = df[df['label'] == 0]['processed_text'].tolist()

        # Validate we got texts
        if not phishing_texts or not legitimate_texts:
            raise ValueError("Failed to extract texts - one or both classes are empty")

        return ' '.join(phishing_texts), ' '.join(legitimate_texts)


def main():
    """Main function to test data preprocessing."""
    logger.info("Starting data preprocessing...")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(max_features=5000, random_state=42)

    # Download dataset
    df = preprocessor.download_dataset()

    # Preprocess
    df = preprocessor.preprocess_dataframe(df)

    # Split for ML models
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = preprocessor.split_for_ml(df, test_size=0.3)

    # Split for transformer models
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = preprocessor.split_for_transformer(df, test_size=0.2)

    # Get texts for word cloud
    phishing_text, legitimate_text = preprocessor.get_texts_by_label(df)

    logger.info("✓ Data preprocessing complete!")
    logger.info(f"✓ Final dataset shape: {df.shape}")
    logger.info(f"✓ Label distribution: {df['label'].value_counts().to_dict()}")

    return df, preprocessor


if __name__ == "__main__":
    df, preprocessor = main()
