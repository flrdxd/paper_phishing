"""
Main Execution Script for Phishing Email Detection - Scikit-Learn Only Version
This version only uses scikit-learn models and avoids PyTorch/Transformers
for compatibility with servers that have deep learning library issues.
"""

import os
import sys
import time
import pandas as pd
import logging

# Import path configuration first
from path_config import PATHS

from data_preprocessing import DataPreprocessor
from models.naive_bayes import NaiveBayesPhishingDetector, train_naive_bayes
from models.dandelion_nb import DandelionNaiveBayesDetector, train_dandelion_nb
from utils.evaluation import (
    compare_models_metrics,
    plot_model_comparison,
    plot_training_time_comparison,
    format_results_table
)
from utils.visualization import (
    generate_phishing_word_cloud,
    generate_legitimate_word_cloud,
    generate_comparative_word_clouds,
    plot_dataset_distribution
)
from utils.results_exporter import (
    export_results_to_text,
    export_results_to_json,
    export_results_to_csv,
    export_all_formats
)

# Set up logging
log_file = os.path.join(PATHS['LOGS_DIR'], 'phishing_detection_sklearn.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PhishingDetectionPipelineSklearn:
    """Pipeline using only scikit-learn models (no deep learning)."""

    def __init__(self, max_features=5000, random_state=42):
        """
        Initialize pipeline.

        Args:
            max_features: Maximum number of TF-IDF features
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.results = {}
        self.total_time = None
        self.PATHS = PATHS

    def load_and_preprocess_data(self, force_download=False):
        """
        Load and preprocess dataset.

        Args:
            force_download: Force re-download of dataset

        Returns:
            tuple: ML data splits
        """
        logger.info("="*60)
        logger.info("PHASE 1: DATA LOADING AND PREPROCESSING")
        logger.info("="*60)

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(max_features=self.max_features, random_state=self.random_state)

        # Download dataset
        df = self.preprocessor.download_dataset(force_download=force_download)

        # Preprocess data
        df = self.preprocessor.preprocess_dataframe(df)

        # Plot dataset distribution
        plot_dataset_distribution(df, save_path=os.path.join(PATHS['PLOTS_DIR'], 'dataset_distribution.png'))

        # Split data for ML models
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = self.preprocessor.split_for_ml(df, test_size=0.3)

        # Get texts for word clouds
        phishing_text, legitimate_text = self.preprocessor.get_texts_by_label(df)

        # Generate word clouds
        logger.info("\nGenerating word clouds...")
        generate_comparative_word_clouds(
            phishing_text, legitimate_text,
            save_path=os.path.join(PATHS['PLOTS_DIR'], 'word_clouds_comparison.png')
        )
        generate_phishing_word_cloud(
            phishing_text,
            save_path=os.path.join(PATHS['PLOTS_DIR'], 'phishing_word_cloud.png')
        )
        generate_legitimate_word_cloud(
            legitimate_text,
            save_path=os.path.join(PATHS['PLOTS_DIR'], 'legitimate_word_cloud.png')
        )

        logger.info("Data preprocessing complete!")

        return X_train_ml, X_test_ml, y_train_ml, y_test_ml

    def train_naive_bayes_model(self, X_train, X_test, y_train, y_test):
        """
        Train Naive Bayes model.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels

        Returns:
            dict: Model results
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2A: TRAINING NAIVE BAYES MODEL")
        logger.info("="*60)

        # Initialize and train model
        nb_detector = NaiveBayesPhishingDetector(alpha=1.0, fit_prior=True, random_state=self.random_state)
        nb_detector.train(X_train, y_train)

        # Evaluate model
        metrics = nb_detector.evaluate(X_test, y_test)

        # Save model
        model_path = os.path.join(self.PATHS['MODELS_DIR'], 'naive_bayes_model.pkl')
        nb_detector.save_model(model_path)

        logger.info("\n" + "-"*50)
        logger.info("NAIVE BAYES RESULTS")
        logger.info("-"*50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"\nTraining Time:    {metrics['training_time']:.2f} seconds")
        logger.info(f"Inference Time:   {metrics['inference_time']:.2f} seconds")
        logger.info("-"*50)

        self.models['naive_bayes'] = nb_detector
        return metrics

    def train_dandelion_nb_model(self, X_train, X_test, y_train, y_test, population_size=20, max_iter=50):
        """
        Train Dandelion-optimized Naive Bayes model.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            population_size: Population size for optimization
            max_iter: Maximum iterations for optimization

        Returns:
            dict: Model results
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2B: TRAINING DANDELION-OPTIMIZED NAIVE BAYES")
        logger.info("="*60)

        # Initialize and train model
        dandelion_nb = DandelionNaiveBayesDetector(
            population_size=population_size,
            max_iter=max_iter,
            random_state=self.random_state
        )
        dandelion_nb.train(X_train, y_train)

        # Evaluate model
        metrics = dandelion_nb.evaluate(X_test, y_test)

        # Save model
        model_path = os.path.join(self.PATHS['MODELS_DIR'], 'dandelion_nb_model.pkl')
        dandelion_nb.save_model(model_path)

        logger.info("\n" + "-"*50)
        logger.info("DANDELION-OPTIMIZED NAIVE BAYES RESULTS")
        logger.info("-"*50)
        logger.info(f"Optimized Parameters:")
        logger.info(f"  alpha = {metrics['alpha']:.4f}")
        logger.info(f"  fit_prior = {metrics['fit_prior']}")
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"\nTiming:")
        logger.info(f"  Optimization Time: {metrics['optimization_time']:.2f} seconds")
        logger.info(f"  Training Time:      {metrics['training_time']:.2f} seconds")
        logger.info(f"  Total Time:         {metrics['total_time']:.2f} seconds")
        logger.info("-"*50)

        self.models['dandelion_nb'] = dandelion_nb
        return metrics

    def run_pipeline(self, population_size=20, max_iter=50):
        """
        Run complete pipeline (scikit-learn models only).

        Args:
            population_size: Population size for Dandelion optimization
            max_iter: Maximum iterations for Dandelion optimization

        Returns:
            dict: All results
        """
        start_time = time.time()

        # Phase 1: Data loading and preprocessing
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = self.load_and_preprocess_data()

        # Phase 2: Train models
        self.results['naive_bayes'] = self.train_naive_bayes_model(
            X_train_ml, X_test_ml, y_train_ml, y_test_ml
        )

        self.results['dandelion_nb'] = self.train_dandelion_nb_model(
            X_train_ml, X_test_ml, y_train_ml, y_test_ml,
            population_size=population_size, max_iter=max_iter
        )

        # Phase 3: Compare results
        self.compare_and_visualize_results()

        # Phase 4: Save results
        self.save_results()

        self.total_time = time.time() - start_time

        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total execution time: {self.total_time:.2f} seconds ({self.total_time / 60:.2f} minutes)")

        return self.results

    def compare_and_visualize_results(self):
        """Compare all models and generate visualizations."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: MODEL COMPARISON AND VISUALIZATION")
        logger.info("="*60)

        # Prepare model names mapping
        model_names = {
            'naive_bayes': 'Naive Bayes',
            'dandelion_nb': 'NB + Dandelion'
        }

        # Create comparison data
        comparison_data = {}
        for key, name in model_names.items():
            if key in self.results:
                comparison_data[name] = self.results[key]

        # Create comparison DataFrame
        comparison_df = compare_models_metrics(comparison_data)

        # Display results table
        logger.info("\n" + "-"*60)
        logger.info("MODEL COMPARISON TABLE")
        logger.info("-"*60)
        logger.info(f"\n{format_results_table(comparison_df)}")
        logger.info("-"*60)

        # Generate visualizations
        logger.info("\nGenerating visualizations...")

        # Model comparison bar chart
        plot_model_comparison(comparison_df, save_path=os.path.join(self.PATHS['PLOTS_DIR'], 'model_comparison_sklearn.png'))

        # Training time comparison
        plot_training_time_comparison(comparison_df, save_path=os.path.join(self.PATHS['PLOTS_DIR'], 'training_time_comparison_sklearn.png'))

        logger.info(f"Visualizations saved to {self.PATHS['PLOTS_DIR']}")

    def save_results(self):
        """Save results to multiple formats (TXT, JSON, CSV)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: SAVING RESULTS")
        logger.info("="*60)

        # Export to all formats
        export_results_to_text(self.results, os.path.join(self.PATHS['RESULTS_DIR'], 'results_summary_sklearn.txt'))
        export_results_to_json(self.results, os.path.join(self.PATHS['RESULTS_DIR'], 'model_results_sklearn.json'))
        export_results_to_csv(self.results, os.path.join(self.PATHS['RESULTS_DIR'], 'model_results_sklearn.csv'))

        logger.info(f"\nAll results saved to {self.PATHS['RESULTS_DIR']}/")
        logger.info("  - results_summary_sklearn.txt (human-readable)")
        logger.info("  - model_results_sklearn.json (machine-readable)")
        logger.info("  - model_results_sklearn.csv (spreadsheet-compatible)")


def main():
    """Main function to run the sklearn-only pipeline."""
    logger.info("="*60)
    logger.info("PHISHING EMAIL DETECTION PIPELINE (SCIKIT-LEARN ONLY)")
    logger.info("Based on: Optimizing Phishing Detection: Comparative")
    logger.info("Analysis of Lightweight Machine Learning Models")
    logger.info("="*60)

    # Initialize pipeline
    pipeline = PhishingDetectionPipelineSklearn(max_features=5000, random_state=42)

    # Run pipeline
    results = pipeline.run_pipeline(population_size=20, max_iter=50)

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    model_names = ['naive_bayes', 'dandelion_nb']
    model_display_names = ['Naive Bayes', 'NB + Dandelion']

    for model_key, model_name in zip(model_names, model_display_names):
        if model_key in results:
            metrics = results[model_key]
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("Pipeline execution complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()