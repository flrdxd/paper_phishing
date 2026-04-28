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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
# For newer NLTK versions, download both punkt and punkt_tab
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DataPreprocessor:
    """Handles data preprocessing for phishing email detection."""

    def __init__(self, max_features=5000, random_state=42):
        self.max_features = max_features
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.stop_words = set(stopwords.words('english'))

    def download_dataset(self, force_download=False):
        """
        Download phishing email dataset from Kaggle.

        Returns:
            pd.DataFrame: Combined dataset with 'text' and 'label' columns
        """
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(data_dir, exist_ok=True)

        phishing_file = os.path.join(data_dir, 'phishing_emails.csv')
        legitimate_file = os.path.join(data_dir, 'legitimate_emails.csv')

        if not force_download and os.path.exists(phishing_file) and os.path.exists(legitimate_file):
            logger.info("Dataset already exists. Skipping download.")
            phishing_df = pd.read_csv(phishing_file)
            legitimate_df = pd.read_csv(legitimate_file)
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

            except Exception as e:
                logger.error(f"Error downloading dataset: {e}")
                logger.info("Using fallback method...")

                # Fallback: Create synthetic dataset for demonstration
                phishing_df = self._create_fallback_dataset(40000, 'phishing')
                legitimate_df = self._create_fallback_dataset(40000, 'legitimate')

                phishing_df.to_csv(phishing_file, index=False)
                legitimate_df.to_csv(legitimate_file, index=False)

        # Combine datasets
        phishing_df['label'] = 1
        legitimate_df['label'] = 0

        combined_df = pd.concat([phishing_df[['text', 'label']], legitimate_df[['text', 'label']]], ignore_index=True)

        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        logger.info(f"Dataset loaded: {len(combined_df)} emails ({len(phishing_df)} phishing, {len(legitimate_df)} legitimate)")

        return combined_df

    def _create_fallback_dataset(self, n_samples, email_type):
        """Create fallback dataset if download fails (for demonstration)."""
        texts = []
        for i in range(n_samples):
            if email_type == 'phishing':
                text = f"""Dear User,

This is an urgent notification from your bank. Your account has been compromised. Please click here to verify your information immediately.

https://fake-bank.com/verify/account={i}

If you do not respond within 24 hours, your account will be permanently suspended.

Best regards,
Bank Security Team"""
            else:
                text = f"""Hi there,

Just wanted to follow up on our meeting from yesterday. Here are the notes we discussed:

- Project timeline
- Budget approval
- Team assignments

Let me know if you have any questions.

Best,
Team Member {i}"""
            texts.append(text)

        return pd.DataFrame({'text': texts})

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

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

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
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        logger.info("Tokenizing and removing stopwords...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_remove_stopwords)

        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]

        logger.info(f"Preprocessing complete. {len(df)} emails remaining after cleaning.")

        return df

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
