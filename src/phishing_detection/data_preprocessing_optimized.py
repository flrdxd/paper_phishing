"""
Optimized Data Preprocessing for Phishing Email Detection

This module provides maximum optimization for ML models:
- Advanced data augmentation
- Optimized TF-IDF features
- Enhanced text cleaning
- Efficient memory management
- Data quality improvements
"""

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Download wordnet (for lemmatization)
try:
    nltk.data.find('corpora/wordnet')
    logger.info("NLTK wordnet already downloaded")
except LookupError:
    logger.info("Downloading NLTK wordnet...")
    nltk.download('wordnet', quiet=False)
    logger.info("NLTK wordnet downloaded successfully")


class OptimizedDataPreprocessor:
    """
    Optimized data preprocessor with maximum performance enhancements.
    """

    def __init__(self, max_features=10000, random_state=42, use_augmentation=True):
        """
        Initialize optimized preprocessor.

        Args:
            max_features: Maximum TF-IDF features (increased for better performance)
            random_state: Random state for reproducibility
            use_augmentation: Whether to use data augmentation
        """
        self.max_features = max_features  # Increased from 5000 to 10000
        self.random_state = random_state
        self.use_augmentation = use_augmentation
        self.tfidf_vectorizer = None
        self.stop_words = set(stopwords.words('english'))

        # Additional optimization parameters
        self.min_df = 3  # Lower minimum document frequency
        self.max_df = 0.85  # Higher maximum document frequency
        self.ngram_range = (1, 3)  # Include trigrams for better patterns

        logger.info(f"Optimized preprocessor initialized: max_features={max_features}, use_augmentation={use_augmentation}")

    def advanced_clean_text(self, text):
        """
        Advanced text cleaning for maximum model performance.

        Args:
            text: Raw text string

        Returns:
            str: Highly cleaned text
        """
        if pd.isna(text):
            return ""

        text = str(text)

        # Remove HTML tags completely
        text = BeautifulSoup(text, 'lxml').get_text()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove phone numbers
        text = re.sub(r'\+?\d{3}[-.]?\d{3}[-.]?\d{4}[-.]?\d{4}', '', text)

        # Remove currency symbols
        text = re.sub(r'[$£€¥₹₽₩₪]', '', text)

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\-,?!]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Convert to lowercase
        text = text.lower()

        return text

    def augment_text(self, text):
        """
        Apply text augmentation for better generalization.

        Args:
            text: Cleaned text string

        Returns:
            list: Augmented texts
        """
        if not self.use_augmentation:
            return [text]

        try:
            # Use NLTK synonym replacement
            from nltk.corpus import wordnet
            from nltk.tokenize import word_tokenize

            syn = wordnet.synsets
            words = word_tokenize(text)

            augmented_texts = [text]  # Original

            for word in words:
                if len(word) > 4:  # Only augment longer words
                    word_synsets = syn(word)
                    if word_synsets:
                        # Get synonyms
                        synonyms = []
                        for synset in word_synsets:
                            for lemma in synset.lemmas():
                                if lemma.name() != word.lower():
                                    synonyms.append(lemma.name())

                        # Select random synonym
                        if synonyms:
                            new_word = np.random.choice(synonyms)
                            augmented_text = text.replace(word, new_word)
                            if augmented_text != text:
                                augmented_texts.append(augmented_text)

            return augmented_texts[:3]  # Limit to 3 augmentations

        except Exception as e:
            logger.warning(f"Text augmentation failed: {e}")
            return [text]

    def tokenize_advanced(self, text):
        """
        Advanced tokenization with fallback.

        Args:
            text: Cleaned text string

        Returns:
            list: Tokens
        """
        try:
            tokens = nltk.word_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}")
            # Fallback to simple whitespace tokenization
            tokens = text.split()

        return tokens

    def remove_stopwords_advanced(self, tokens):
        """
        Advanced stopword removal.

        Args:
            tokens: List of tokens

        Returns:
            list: Filtered tokens
        """
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words and len(token) > 2]
        return filtered_tokens

    def preprocess_dataframe_optimized(self, df):
        """
        Optimized preprocessing with augmentation and quality improvements.

        Args:
            df: DataFrame with 'text' and 'label' columns

        Returns:
            pd.DataFrame: Optimized and preprocessed DataFrame
        """
        logger.info("Advanced text cleaning...")
        df['cleaned_text'] = df['text'].apply(self.advanced_clean_text)

        logger.info("Advanced tokenization...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_advanced)

        logger.info("Advanced stopword removal...")
        df['filtered_tokens'] = df['tokens'].apply(self.remove_stopwords_advanced)

        logger.info("Text length calculation...")
        df['text_length'] = df['filtered_tokens'].apply(lambda x: len(x))

        # Remove very short or very long texts (quality control)
        min_length = 10
        max_length = 1000
        df = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]

        # Convert to processed text
        df['processed_text'] = df['filtered_tokens'].apply(lambda x: ' '.join(x))

        # Apply augmentation if enabled
        if self.use_augmentation:
            logger.info("Applying text augmentation...")
            augmented_rows = []
            for idx, row in df.iterrows():
                augmented_texts = self.augment_text(row['processed_text'])[1:]
                for aug_text in augmented_texts:
                    new_row = row.copy()
                    new_row['processed_text'] = aug_text
                    augmented_rows.append(new_row)

            # Combine original and augmented
            augmented_df = pd.DataFrame(augmented_rows)
            if not augmented_df.empty:
                df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(f"Augmentation complete: {len(augmented_df)} samples")

        # Final data quality check
        logger.info("Data quality validation...")
        df = df.dropna(subset=['processed_text', 'label'])

        logger.info(f"Preprocessing complete: {len(df)} samples remaining")
        return df

    def tfidf_vectorize_optimized(self, texts, fit=True):
        """
        Optimized TF-IDF vectorization.

        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizer

        Returns:
            np.ndarray: TF-IDF vectors
        """
        if fit or self.tfidf_vectorizer is None:
            logger.info(f"Creating optimized TF-IDF vectorizer: max_features={self.max_features}")

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range,
                sublinear_tf=True,  # Sublinear TF scaling for better performance
                use_idf=True,
                norm='l2',  # L2 normalization
                stop_words='english'
            )

            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF fitted. Shape: {tfidf_matrix.shape}")

            # Feature analysis
            n_features = tfidf_matrix.shape[1]
            avg_nnz = tfidf_matrix.nnz / n_features
            logger.info(f"Features: {n_features}, Avg non-zero per sample: {avg_nnz:.2f}")

            return tfidf_matrix
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            logger.info(f"TF-IDF transformed. Shape: {tfidf_matrix.shape}")
            return tfidf_matrix

    def split_for_ml_optimized(self, df, test_size=0.2):
        """
        Optimized data splitting with stratification.

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Optimized data splitting (test_size={test_size})...")

        X = df['processed_text']
        y = df['label']

        # Stratified split to maintain label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # TF-IDF vectorization
        X_train_tfidf = self.tfidf_vectorize_optimized(X_train.values, fit=True)
        X_test_tfidf = self.tfidf_vectorize_optimized(X_test.values, fit=False)

        logger.info(f"Data splits: Train={X_train_tfidf.shape[0]}, Test={X_test_tfidf.shape[0]}")
        logger.info(f"Label distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Label distribution - Test: {y_test.value_counts().to_dict()}")

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def split_for_transformer_optimized(self, df, test_size=0.2):
        """
        Optimized data splitting for transformer models.

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Optimized transformer data splitting (test_size={test_size})...")

        X = df['cleaned_text']  # Use cleaned text (with stopwords) for transformers
        y = df['label']

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Data splits: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Label distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Label distribution - Test: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test


def main():
    """Main function to test optimized preprocessing."""
    logger.info("Testing optimized data preprocessing...")

    # Test with synthetic data
    preprocessor = OptimizedDataPreprocessor(max_features=10000, random_state=42, use_augmentation=False)

    # Create larger, more realistic test dataset
    test_texts = [
        "Dear Customer, urgent notification your account compromised click verify information immediately",
        "Hi team, just following up on our meeting notes we discussed budget approval team assignments",
        "Click here to claim your prize winner selected you have 24 hours respond",
        "Meeting rescheduled due to technical issues please check your email for updates",
        "Your account has been suspended please update your payment information immediately",
        "Please find attached the quarterly financial report for your review and approval",
        "Congratulations you have won a million dollars click here to claim your prize now",
        "Looking forward to our discussion tomorrow about the project milestones and deliverables",
        "Verify your identity now or your account will be permanently closed within 24 hours",
        "I wanted to share some thoughts about the recent market trends and potential opportunities",
        "Security alert unusual login detected from unrecognized device please confirm your activity",
        "Thank you for your continued support and partnership throughout this fiscal year",
        "Limited time offer exclusive discount just for you act now before it expires",
        "The training session has been moved to Conference Room B at 3 PM today",
        "Your bank account has been compromised please click this link to secure your funds",
        "Please review the attached documents and let me know if you have any questions",
        "Urgent action required your subscription is about to expire renew now to continue service",
        "I hope this email finds you well and looking forward to catching up soon",
        "You have been selected for a special promotion don't miss out on this amazing opportunity",
        "The quarterly results show significant growth in all key performance indicators"
    ] * 10  # Repeat to get more samples
    test_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10  # Phishing, Legitimate alternating

    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

    # Preprocess
    processed_df = preprocessor.preprocess_dataframe_optimized(test_df)

    # Test TF-IDF
    X_train, X_test, y_train, y_test = preprocessor.split_for_ml_optimized(processed_df)

    logger.info("Optimized preprocessing test complete!")
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    return processed_df, preprocessor


if __name__ == "__main__":
    df, preprocessor = main()
