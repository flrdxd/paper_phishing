"""
Data Preprocessing Module for Phishing Email Detection

This module handles:
- Dataset download from Kaggle
- Text cleaning (HTML removal, special characters)
- Tokenization and stopwords removal
- TF-IDF vectorization
- Train/test split for ML and transformer models
"""

import os
import re
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
from path_config import PATHS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
# For newer NLTK versions, download both punkt and punkt_tab
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

    def download_dataset(self, force_download=False):
        """
        Download phishing email dataset from Kaggle.

        Returns:
            pd.DataFrame: Combined dataset with 'text' and 'label' columns
        """
        data_dir = PATHS['DATA_DIR']
        os.makedirs(data_dir, exist_ok=True)

        phishing_file = os.path.join(data_dir, 'phishing_emails.csv')
        legitimate_file = os.path.join(data_dir, 'legitimate_emails.csv')

        if not force_download and os.path.exists(phishing_file) and os.path.exists(legitimate_file):
            logger.info("Dataset already exists. Skipping download.")
            logger.warning("⚠️  WARNING: Using cached dataset. If accuracy is suspicious, delete cache files and re-download.")
            phishing_df = pd.read_csv(phishing_file)
            legitimate_df = pd.read_csv(legitimate_file)

            # CRITICAL: Check if cached data is synthetic/fallback
            if self._is_synthetic_dataset(phishing_df, legitimate_df):
                logger.error("❌ CRITICAL: Cached dataset appears to be SYNTHETIC/FALLBACK data!")
                logger.error("❌ This will cause ARTIFICIAL accuracy and meaningless results!")
                logger.error("❌ Delete the cached files and ensure Kaggle download works:")
                logger.error(f"   rm {phishing_file}")
                logger.error(f"   rm {legitimate_file}")
                raise ValueError("Synthetic dataset detected. Delete cache files and re-download real data.")
        else:
            logger.info("Downloading dataset from Kaggle...")
            try:
                # Download phishing email dataset
                path = kagglehub.dataset_download("subhajournal/phishingemails")
                dataset_path = os.path.join(path, "Phishing_Email.csv")
                phishing_df = pd.read_csv(dataset_path)

                # For legitimate emails, we'll use Enron or similar
                # Using a publicly available spam dataset that includes ham emails
                path2 = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
                spam_path = os.path.join(path2, "spam.csv")
                spam_df = pd.read_csv(spam_path, encoding='latin-1')

                # Filter ham messages
                legitimate_df = spam_df[spam_df['v1'] == 'ham'].sample(n=len(phishing_df), random_state=self.random_state)

                # Rename columns to match our format
                phishing_df = phishing_df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})
                phishing_df['label'] = phishing_df['label'].map({'Phishing Email': 1, 'Safe Email': 0})

                # For legitimate dataframe, create proper format
                legitimate_df = legitimate_df.rename(columns={'v2': 'text'})
                legitimate_df['label'] = 0

                # Save to files
                phishing_df.to_csv(phishing_file, index=False)
                legitimate_df.to_csv(legitimate_file, index=False)

                logger.info(f"Dataset downloaded and saved to {data_dir}")
                logger.info("✓ Real dataset from Kaggle downloaded successfully")

            except Exception as e:
                logger.error(f"❌ CRITICAL: Error downloading dataset: {e}")
                logger.error("❌ Cannot proceed without real dataset!")
                logger.error("❌ Fallback synthetic datasets are DISABLED to prevent artificial accuracy.")
                logger.error("❌ Please fix the Kaggle download issue:")
                logger.error("   1. Check internet connection")
                logger.error("   2. Verify Kaggle API credentials")
                logger.error("   3. Ensure kagglehub is properly installed")
                raise RuntimeError(f"Failed to download real dataset: {e}. Fallback disabled to prevent artificial accuracy.")

        # Combine datasets
        phishing_df['label'] = 1
        legitimate_df['label'] = 0

        combined_df = pd.concat([phishing_df[['text', 'label']], legitimate_df[['text', 'label']]], ignore_index=True)

        # Remove duplicate samples (quality check)
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        duplicates_removed = initial_count - len(combined_df)

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate samples for data quality")

        # Basic data quality check
        min_length = combined_df['text'].str.len().min()
        max_length = combined_df['text'].str.len().max()
        avg_length = combined_df['text'].str.len().mean()

        logger.info(f"Text length statistics - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.1f}")

        # Warn about potential data quality issues
        if min_length < 10:
            logger.warning("WARNING: Very short samples detected - potential data quality issue")

        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        logger.info(f"Dataset loaded: {len(combined_df)} emails ({len(phishing_df)} phishing, {len(legitimate_df)} legitimate)")

        return combined_df

    def _is_synthetic_dataset(self, phishing_df, legitimate_df):
        """
        CRITICAL: Detect if dataset is synthetic/fallback.

        Synthetic datasets have extremely low diversity and cause artificial accuracy.
        This check prevents training on meaningless data.

        Args:
            phishing_df: Phishing emails DataFrame
            legitimate_df: Legitimate emails DataFrame

        Returns:
            bool: True if dataset appears synthetic
        """
        # Check 1: Extremely low text diversity
        phishing_diversity = phishing_df['text'].nunique() / len(phishing_df)
        legitimate_diversity = legitimate_df['text'].nunique() / len(legitimate_df)

        if phishing_diversity < 0.2 or legitimate_diversity < 0.2:
            logger.error(f"Synthetic dataset detected: Low text diversity (phishing: {phishing_diversity:.2%}, legitimate: {legitimate_diversity:.2%})")
            return True

        # Check 2: Template repetition (most samples start with same phrase)
        phishing_first_words = phishing_df['text'].str.split().str[:3].apply(' '.join)
        legitimate_first_words = legitimate_df['text'].str.split().str[:3].apply(' '.join)

        phishing_top_ratio = phishing_first_words.value_counts().iloc[0] / len(phishing_df)
        legitimate_top_ratio = legitimate_first_words.value_counts().iloc[0] / len(legitimate_df)

        if phishing_top_ratio > 0.7 or legitimate_top_ratio > 0.7:
            logger.error(f"Synthetic dataset detected: High template repetition (phishing: {phishing_top_ratio:.2%}, legitimate: {legitimate_top_ratio:.2%})")
            return True

        # Check 3: Suspicious vocabulary size
        all_text = ' '.join(phishing_df['text'].astype(str)) + ' ' + ' '.join(legitimate_df['text'].astype(str))
        unique_words = len(set(all_text.lower().split()))

        if unique_words < 200:
            logger.error(f"Synthetic dataset detected: Extremely low vocabulary ({unique_words} unique words)")
            return True

        return False

    def _create_fallback_dataset(self, n_samples, email_type):
        """
        DEPRECATED: Create fallback dataset if download fails.

        ⚠️  WARNING: This method is DISABLED to prevent artificial accuracy.
        Synthetic datasets cause meaningless results and should never be used.

        Args:
            n_samples: Number of samples to generate
            email_type: Type of email ('phishing' or 'legitimate')

        Raises:
            RuntimeError: Always raises to prevent synthetic data usage
        """
        raise RuntimeError(
            "Fallback dataset generation is DISABLED.\n"
            "Synthetic datasets cause artificial accuracy and are meaningless for research.\n"
            "Please fix the Kaggle download issue instead."
        )

    def clean_text(self, text):
        """
        Clean text by removing HTML, special characters, etc.

        Args:
            text: Raw text string

        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to string
        text = str(text)

        # Remove HTML tags
        text = BeautifulSoup(text, 'lxml').get_text()

        # Extract URL features for phishing detection (instead of removing them)
        url_pattern = r'http\S+|www\S+|https\S+'
        urls = re.findall(url_pattern, text, flags=re.MULTILINE)

        # Feature extraction from URLs
        url_features = {
            'has_url': len(urls) > 0,
            'url_count': len(urls),
            'has_ip_in_url': any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text) for _ in urls),
            'avg_url_length': np.mean([len(url) for url in urls]) if urls else 0,
            'max_url_length': max([len(url) for url in urls]) if urls else 0
        }

        # Check for suspicious TLDs in URLs
        suspicious_tld_count = sum(1 for url in urls if any(url.endswith(tld) for tld in self.suspicious_tlds))

        # Keep URLs for feature extraction, but anonymize them for training
        # This preserves the important signal for phishing detection
        anonymized_text = text
        url_counter = 0
        for url in urls:
            anonymized_url = f"http://domain-{url_counter}.com"
            anonymized_text = anonymized_text.replace(url, anonymized_url)
            url_counter += 1

        # Add URL features to a temporary attribute for later use
        self._temp_url_features = url_features | {'suspicious_tld_count': suspicious_tld_count}

        # Remove email addresses (keep domain info)
        email_pattern = r'\S+@\S+'
        emails = re.findall(email_pattern, anonymized_text)

        # Remove extra email addresses but keep domains for context
        for email in emails:
            # Extract domain from email
            domain = email.split('@')[-1] if '@' in email else ''
            anonymized_text = anonymized_text.replace(email, f"@domain-{len(emails)}.com")

        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', anonymized_text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
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
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            # Fallback to simple whitespace tokenization if NLTK fails
            logger.warning(f"NLTK tokenization failed: {e}. Using simple tokenization.")
            tokens = text.split()

        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_dataframe(self, df):
        """
        Preprocess entire DataFrame.

        Args:
            df: DataFrame with 'text' and 'label' columns

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # CRITICAL: Quality check before preprocessing
        self._quality_check_before_preprocessing(df)

        logger.info("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        logger.info("Tokenizing and removing stopwords...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_remove_stopwords)

        # Add URL features from temporary storage if available
        if hasattr(self, '_temp_url_features'):
            url_features_df = pd.DataFrame([self._temp_url_features] * len(df))
            # Add URL features as columns
            df['has_url'] = url_features_df['has_url']
            df['url_count'] = url_features_df['url_count']
            df['has_ip_in_url'] = url_features_df['has_ip_in_url']
            df['avg_url_length'] = url_features_df['avg_url_length']
            df['max_url_length'] = url_features_df['max_url_length']
            df['suspicious_tld_count'] = url_features_df['suspicious_tld_count']

            logger.info(f"URL features extracted - has_url: {df['has_url'].sum()}/{len(df)} samples")

        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]

        logger.info(f"Preprocessing complete. {len(df)} emails remaining after cleaning.")

        return df

    def _quality_check_before_preprocessing(self, df):
        """
        CRITICAL: Perform quality checks on raw dataset.

        This prevents training on problematic data that would cause
        artificial accuracy or meaningless results.

        Args:
            df: DataFrame to check

        Raises:
            ValueError: If critical quality issues are detected
        """
        logger.info("\n" + "="*60)
        logger.info("DATASET QUALITY CHECK")
        logger.info("="*60)

        # Check 1: Dataset size
        if len(df) < 100:
            raise ValueError(f"Dataset too small: {len(df)} samples. Minimum 100 required.")

        # Check 2: Text diversity
        unique_texts = df['text'].nunique()
        diversity_ratio = unique_texts / len(df)

        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Unique texts: {unique_texts} ({diversity_ratio:.2%})")

        if diversity_ratio < 0.3:
            error_msg = f"CRITICAL: Extremely low text diversity ({diversity_ratio:.2%}). Dataset appears synthetic!"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Check 3: Template repetition
        first_words = df['text'].str.split().str[:3].apply(' '.join)
        top_template_ratio = first_words.value_counts().iloc[0] / len(df)

        logger.info(f"Top template coverage: {top_template_ratio:.2%}")

        if top_template_ratio > 0.5:
            error_msg = f"CRITICAL: High template repetition ({top_template_ratio:.2%}). Dataset appears synthetic!"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Check 4: Label distribution
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            error_msg = "CRITICAL: Dataset contains only one class!"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"Label distribution: {label_counts.to_dict()}")

        # Check 5: Vocabulary size
        all_words = ' '.join(df['text'].astype(str)).lower().split()
        unique_words = len(set(all_words))

        logger.info(f"Vocabulary size: {unique_words} unique words")

        if unique_words < 100:
            error_msg = f"CRITICAL: Extremely low vocabulary ({unique_words} words). Dataset appears synthetic!"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        logger.info("✓ Dataset quality checks passed")
        logger.info("="*60 + "\n")

    def tfidf_vectorize(self, texts, fit=True):
        """
        Convert texts to TF-IDF vectors.

        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizer (True for training data)

        Returns:
            np.ndarray: TF-IDF vectors
        """
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=5,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF vectorization complete. Shape: {tfidf_matrix.shape}")
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            logger.info(f"TF-IDF transformation complete. Shape: {tfidf_matrix.shape}")

        return tfidf_matrix

    def split_for_ml(self, df, test_size=0.3):
        """
        Split data for ML models (Naive Bayes, Dandelion).

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test (TF-IDF vectors and labels)
        """
        logger.info(f"Splitting data for ML models (test_size={test_size})...")

        X = df['processed_text']
        y = df['label']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # TF-IDF vectorization
        X_train_tfidf = self.tfidf_vectorize(X_train, fit=True)
        X_test_tfidf = self.tfidf_vectorize(X_test, fit=False)

        logger.info(f"Training samples: {X_train_tfidf.shape[0]}, Test samples: {X_test_tfidf.shape[0]}")
        logger.info(f"Training labels distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test labels distribution: {y_test.value_counts().to_dict()}")

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def split_for_transformer(self, df, test_size=0.2):
        """
        Split data for transformer models (BERT, DistilBERT).

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test (texts and labels)
        """
        logger.info(f"Splitting data for transformer models (test_size={test_size})...")

        X = df['cleaned_text']  # Use cleaned text (with stopwords) for transformers
        y = df['label']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Training labels distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test labels distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def get_texts_by_label(self, df):
        """
        Get texts separated by label for word cloud generation.

        Args:
            df: Preprocessed DataFrame

        Returns:
            tuple: phishing_texts, legitimate_texts
        """
        phishing_texts = df[df['label'] == 1]['processed_text'].tolist()
        legitimate_texts = df[df['label'] == 0]['processed_text'].tolist()

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

    logger.info("Data preprocessing complete!")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df, preprocessor


if __name__ == "__main__":
    df, preprocessor = main()
