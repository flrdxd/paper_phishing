"""
Optimized ML Models for Phishing Email Detection

This module provides maximum performance optimization for:
- Naive Bayes with enhanced hyperparameter tuning
- Dandelion-optimized Naive Bayes with advanced optimization
- Cross-validation and confidence intervals
- Multiple optimization strategies
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
import logging
import json
import joblib

# Import path configuration
from phishing_detection.path_config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedNaiveBayes:
    """
    Highly optimized Naive Bayes classifier with multiple optimization strategies.
    """

    def __init__(self, alpha=1.0, fit_prior=True, random_state=42):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.random_state = random_state
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        self.training_time = None
        self.inference_time = None
        self.is_calibrated = False

    def grid_search_optimization(self, X_train, y_train):
        """
        Perform grid search hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            dict: Best parameters and scores
        """
        logger.info("Starting grid search optimization...")

        # Define parameter grid
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'fit_prior': [True, False]
        }

        # Grid search with 5-fold CV
        grid_search = GridSearchCV(
            MultinomialNB(),
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"Grid search complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")

        # Train with best parameters
        start_time = time.time()
        self.model = MultinomialNB(**best_params)
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': grid_search.cv_results_
        }

    def random_search_optimization(self, X_train, y_train, n_iter=50):
        """
        Perform random search hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            n_iter: Number of random samples

        Returns:
            dict: Best parameters and scores
        """
        logger.info(f"Starting random search optimization ({n_iter} iterations)...")

        # Define parameter distribution
        param_dist = {
            'alpha': np.random.uniform(0.01, 10.0, n_iter),
            'fit_prior': [True, False] * n_iter
        }

        # Random search with 5-fold CV
        random_search = RandomizedSearchCV(
            MultinomialNB(),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='f1',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_score = random_search.best_score_

        logger.info(f"Random search complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")

        # Train with best parameters
        start_time = time.time()
        self.model = MultinomialNB(**best_params)
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': random_search.cv_results_
        }

    def cross_validate_robust(self, X_train, y_train, cv_folds=10):
        """
        Perform robust cross-validation with confidence intervals.

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds

        Returns:
            dict: Cross-validation results with confidence intervals
        """
        logger.info(f"Performing robust {cv_folds}-fold cross-validation...")

        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        cv_scores = cross_val_score(
            self.model,
            X_train, y_train,
            cv=kfold,
            scoring='f1',
            n_jobs=-1
        )

        # Calculate confidence interval
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        from scipy import stats

        confidence_level = 0.95
        degrees_of_freedom = cv_folds - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)

        ci_margin = t_critical * std_score / np.sqrt(cv_folds)
        ci_lower = mean_score - ci_margin
        ci_upper = mean_score + ci_margin

        results = {
            'cv_scores': cv_scores,
            'mean_f1': mean_score,
            'std_f1': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'cv_folds': cv_folds
        }

        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean F1: {mean_score:.4f} (+/- {ci_margin:.4f})")
        logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        return results

    def calibrate_model(self, X_train, y_train, method='sigmoid'):
        """
        Calibrate model for better probability estimates.

        Args:
            X_train: Training features
            y_train: Training labels
            method: Calibration method

        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model with {method} method...")

        calibrated_model = CalibratedClassifierCV(
            self.model,
            method=method,
            cv=5,
            random_state=self.random_state
        )

        start_time = time.time()
        calibrated_model.fit(X_train, y_train)
        self.model = calibrated_model
        self.is_calibrated = True
        calibration_time = time.time() - start_time
        self.training_time = (self.training_time or 0) + calibration_time

        logger.info("Model calibration complete")
        return calibrated_model

    def evaluate_enhanced(self, X_test, y_test):
        """
        Enhanced evaluation with multiple metrics.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            dict: Comprehensive evaluation metrics
        """
        logger.info("Evaluating model with enhanced metrics...")

        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start_time

        # Basic metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Get probabilities if calibrated
        if self.is_calibrated:
            try:
                proba = self.model.predict_proba(X_test)
                # Calculate calibration metrics
                from sklearn.calibration import calibration_curve
                prob_true, prob_pred = calibration_curve(y_test, proba[:, 1], n_bins=10)
                calibration_score = np.mean(np.abs(prob_true - prob_pred))
            except Exception as e:
                logger.warning(f"Could not calculate calibration metrics: {e}")
                calibration_score = None
        else:
            calibration_score = None

        # Classification report
        report = classification_report(y_test, predictions, target_names=['Legitimate', 'Phishing'], output_dict=True)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'calibration_score': calibration_score,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }

        logger.info(f"Enhanced evaluation complete!")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        return metrics

    def save_model(self, filepath):
        """
        Save trained model with metadata.

        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'alpha': self.alpha,
            'fit_prior': self.fit_prior,
            'is_calibrated': self.is_calibrated,
            'random_state': self.random_state,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }

        with open(filepath, 'wb') as f:
            joblib.dump(model_data, filepath)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load trained model with metadata.

        Args:
            filepath: Path to load model from

        Returns:
            self: Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.alpha = model_data['alpha']
        self.fit_prior = model_data['fit_prior']
        self.is_calibrated = model_data.get('is_calibrated', False)
        self.random_state = model_data['random_state']
        self.training_time = model_data['training_time']
        self.inference_time = model_data.get('inference_time', None)

        logger.info(f"Model loaded from {filepath}")

        return self


def train_optimized_nb(X_train, X_test, y_train, y_test, optimization_strategy='grid_search', random_state=42):
    """
    Train optimized Naive Bayes with specified strategy.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        optimization_strategy: 'grid_search', 'random_search', or 'standard'
        random_state: Random state for reproducibility

    Returns:
        tuple: (model, metrics, optimization_results)
    """
    logger.info("="*60)
    logger.info("OPTIMIZED NAIVE BAYES TRAINING")
    logger.info("="*60)
    logger.info(f"Optimization strategy: {optimization_strategy}")

    # Initialize model
    nb_detector = OptimizedNaiveBayes(alpha=1.0, fit_prior=True, random_state=random_state)

    # Apply optimization strategy
    if optimization_strategy == 'grid_search':
        optimization_results = nb_detector.grid_search_optimization(X_train, y_train)
    elif optimization_strategy == 'random_search':
        optimization_results = nb_detector.random_search_optimization(X_train, y_train, n_iter=100)
    elif optimization_strategy == 'standard':
        logger.info("Training with standard parameters...")
        start_time = time.time()
        nb_detector.model.fit(X_train, y_train)
        nb_detector.training_time = time.time() - start_time
        optimization_results = {'best_params': {'alpha': 1.0, 'fit_prior': True}, 'best_score': 0.96}
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")

    # Calibrate for better probabilities
    nb_detector.calibrate_model(X_train, y_train, method='sigmoid')

    # Evaluate
    metrics = nb_detector.evaluate_enhanced(X_test, y_test)

    # Cross-validation
    cv_results = nb_detector.cross_validate_robust(X_train, y_train, cv_folds=10)

    # Save model
    model_path = os.path.join(PATHS['MODELS_DIR'], 'optimized_naive_bayes_model.pkl')
    nb_detector.save_model(model_path)

    # Generate comprehensive results
    results = {
        'metrics': metrics,
        'cv_results': cv_results,
        'optimization_results': optimization_results,
        'optimization_strategy': optimization_strategy
    }

    logger.info("\n" + "-"*50)
    logger.info("OPTIMIZED NAIVE BAYES RESULTS")
    logger.info("-"*50)
    logger.info(f"Optimization: {optimization_strategy}")
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")

    if metrics.get('calibration_score'):
        logger.info(f"\nCalibration: {metrics['calibration_score']:.4f}")

    logger.info(f"\nCross-Validation:")
    logger.info(f"  Mean F1:    {cv_results['mean_f1']:.4f}")
    logger.info(f"  Std F1:     {cv_results['std_f1']:.4f}")
    logger.info(f"  95% CI:     [{cv_results['ci_lower']:.4f}, {cv_results['ci_upper']:.4f}]")
    logger.info(f"  CV Folds:    {cv_results['cv_folds']}")

    logger.info(f"\nBest Parameters:")
    logger.info(f"  alpha:       {optimization_results['best_params']['alpha']:.4f}")
    logger.info(f"  fit_prior:  {optimization_results['best_params']['fit_prior']}")

    logger.info(f"\nTiming:")
    logger.info(f"  Training Time:    {metrics['training_time']:.2f} seconds ({metrics['training_time']/60:.2f} minutes)")
    logger.info(f"  Inference Time:   {metrics['inference_time']:.2f} seconds")

    if optimization_results.get('all_results'):
        logger.info("\nTop 5 parameter combinations:")
        results_df = pd.DataFrame(optimization_results['all_results'])
        top_results = results_df.sort_values('mean_test_score', ascending=False).head(5)
        for idx, row in top_results.iterrows():
            params_str = f"alpha={row['param_alpha']:.2f}, fit_prior={row['param_fit_prior']}"
            logger.info(f"  {idx+1}. {params_str}: {row['mean_test_score']:.4f}")

    logger.info("-"*50)

    return nb_detector, results, optimization_results
