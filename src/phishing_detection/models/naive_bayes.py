"""
Naive Bayes Model for Phishing Email Detection

This module implements Multinomial Naive Bayes with TF-IDF features.
Includes training, evaluation, and cross-validation.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

# Import path configuration
from phishing_detection.path_config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NaiveBayesPhishingDetector:
    """Naive Bayes classifier for phishing email detection."""

    def __init__(self, alpha=1.0, fit_prior=True, random_state=42):
        """
        Initialize Naive Bayes classifier.

        Args:
            alpha: Laplace smoothing parameter
            fit_prior: Whether to learn class prior probabilities
            random_state: Random state for reproducibility
        """
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.random_state = random_state
        self.training_time = None
        self.inference_time = None

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes model.

        Args:
            X_train: Training features (TF-IDF vectors)
            y_train: Training labels

        Returns:
            self: Trained model
        """
        logger.info(f"Training Naive Bayes with alpha={self.alpha}, fit_prior={self.fit_prior}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        logger.info(f"Training complete in {self.training_time:.2f} seconds")

        return self

    def predict(self, X_test):
        """
        Make predictions on test data.

        Args:
            X_test: Test features (TF-IDF vectors)

        Returns:
            np.ndarray: Predictions
        """
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start_time

        logger.info(f"Prediction complete in {self.inference_time:.2f} seconds ({len(predictions)} samples)")

        return predictions

    def predict_proba(self, X_test):
        """
        Get prediction probabilities.

        Args:
            X_test: Test features (TF-IDF vectors)

        Returns:
            np.ndarray: Prediction probabilities
        """
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features (TF-IDF vectors)
            y_test: True labels

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model...")

        predictions = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Classification report
        report = classification_report(y_test, predictions, target_names=['Legitimate', 'Phishing'])

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Calculate security-critical metrics
        tn, fp, fn, tp = cm.ravel()

        # False Positive Rate (FPR) - legitimate emails incorrectly blocked
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # False Negative Rate (FNR) - phishing emails incorrectly allowed
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # True Positive Rate (TPR) - phishing correctly detected
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # True Negative Rate (TNR) - legitimate correctly allowed
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        # For AUC-ROC (simplified version for Naive Bayes)
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_scores = self.model.predict_proba(X_test)[:, 1]  # Probability of phishing class
            auc_roc = roc_auc_score(y_test, y_scores)
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            auc_roc = None
            y_scores = None

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'confusion_matrix': cm,
            'classification_report': report,
            'security_metrics': {
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'true_positive_rate': tpr,
                'true_negative_rate': tnr
            },
            'auc_roc': auc_roc
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"False Positive Rate: {fpr:.4f} ({fpr*100:.1f}%)")
        logger.info(f"False Negative Rate: {fnr:.4f} ({fnr*100:.1f}%)")

        return metrics

    def cross_validate(self, X, y, cv=5):
        """
        Perform k-fold cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")

        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X, y, cv=kfold, scoring='f1', n_jobs=-1
        )

        # Calculate confidence interval
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        ci_lower = mean_score - 1.96 * (std_score / np.sqrt(cv))
        ci_upper = mean_score + 1.96 * (std_score / np.sqrt(cv))

        cv_results = {
            'mean_f1': mean_score,
            'std_f1': std_score,
            'f1_scores': cv_scores,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_range': (ci_lower, ci_upper)
        }

        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean F1: {mean_score:.4f} (+/- {std_score:.4f})")
        logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

        return cv_results

    def save_model(self, filepath):
        """
        Save trained model to file.

        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'alpha': self.alpha,
            'fit_prior': self.fit_prior,
            'random_state': self.random_state,
            'training_time': self.training_time
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load trained model from file.

        Args:
            filepath: Path to load the model from

        Returns:
            self: Loaded model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.alpha = model_data['alpha']
        self.fit_prior = model_data['fit_prior']
        self.random_state = model_data['random_state']
        self.training_time = model_data['training_time']

        logger.info(f"Model loaded from {filepath}")

        return self


def train_naive_bayes(X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Convenience function to train and evaluate Naive Bayes model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        tuple: Trained model and evaluation metrics
    """
    if X_train is None or X_test is None or y_train is None or y_test is None:
        # If data not provided, load and preprocess
        from phishing_detection.data_preprocessing import DataPreprocessor

        logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor(max_features=5000, random_state=42)
        df = preprocessor.download_dataset()
        df = preprocessor.preprocess_dataframe(df)
        X_train, X_test, y_train, y_test = preprocessor.split_for_ml(df, test_size=0.3)

    # Initialize and train model
    nb_detector = NaiveBayesPhishingDetector(alpha=1.0, fit_prior=True, random_state=42)
    nb_detector.train(X_train, y_train)

    # Evaluate model
    metrics = nb_detector.evaluate(X_test, y_test)

    # Cross-validation
    cv_results = nb_detector.cross_validate(X_train, y_train, cv=5)

    # Save model
    model_path = os.path.join(PATHS['MODELS_DIR'], 'naive_bayes_model.pkl')
    nb_detector.save_model(model_path)

    logger.info("\n" + "="*50)
    logger.info("NAIVE BAYES MODEL RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"\nTraining Time:    {metrics['training_time']:.2f} seconds")
    logger.info(f"Inference Time:   {metrics['inference_time']:.2f} seconds")
    logger.info(f"\nCross-Validation Mean F1: {cv_results['mean_f1']:.4f}")
    logger.info(f"95% CI: [{cv_results['ci_lower']:.4f}, {cv_results['ci_upper']:.4f}]")
    logger.info("="*50)

    return nb_detector, metrics, cv_results


if __name__ == "__main__":
    model, metrics, cv_results = train_naive_bayes()
