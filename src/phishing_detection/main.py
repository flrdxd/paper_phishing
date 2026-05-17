"""
Main Execution Script for Phishing Email Detection

This script orchestrates the complete pipeline:
1. Load and preprocess data
2. Train all four models
3. Evaluate each model
4. Generate visualizations
5. Compare results
6. Save trained models
"""

import os
import time
import json
import platform
import pandas as pd
import logging

from phishing_detection.path_config import PATHS
from phishing_detection.data_preprocessing import DataPreprocessor
from phishing_detection.models.naive_bayes import NaiveBayesPhishingDetector
from phishing_detection.models.dandelion_nb import DandelionNaiveBayesDetector
from phishing_detection.models.bert_model import BERTPhishingDetector
from phishing_detection.models.distilbert_model import DistilBERTPhishingDetector
from phishing_detection.utils.evaluation import (
    compare_models_metrics,
    plot_model_comparison,
    plot_training_time_comparison,
    format_results_table
)
from phishing_detection.utils.visualization import (
    generate_phishing_word_cloud,
    generate_legitimate_word_cloud,
    generate_comparative_word_clouds,
    plot_learning_curve,
    plot_training_losses,
    plot_model_comparison_radar,
    plot_accuracy_vs_efficiency,
    plot_dataset_distribution
)
from phishing_detection.utils.results_exporter import (
    export_results_to_text,
    export_results_to_json,
    export_results_to_csv,
    export_all_formats
)

# Set up logging with location-independent path
log_file = os.path.join(PATHS['LOGS_DIR'], 'phishing_detection.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PhishingDetectionPipeline:
    """Main pipeline for phishing email detection."""

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

    def load_and_preprocess_data(self, force_download=False):
        """
        Load and preprocess dataset.

        Args:
            force_download: Force re-download of dataset

        Returns:
            tuple: ML and transformer data splits
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

        # Split data for transformer models
        X_train_tf, X_test_tf, y_train_tf, y_test_tf = self.preprocessor.split_for_transformer(df, test_size=0.2)

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

        return (X_train_ml, X_test_ml, y_train_ml, y_test_ml,
                X_train_tf, X_test_tf, y_train_tf, y_test_tf)

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
        model_path = os.path.join(PATHS['MODELS_DIR'], 'naive_bayes_model.pkl')
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
        model_path = os.path.join(PATHS['MODELS_DIR'], 'dandelion_nb_model.pkl')
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

    def train_bert_model(self, X_train, X_test, y_train, y_test):
        """
        Train BERT model.

        Args:
            X_train: Training texts
            X_test: Test texts
            y_train: Training labels
            y_test: Test labels

        Returns:
            dict: Model results
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2C: TRAINING BERT MODEL")
        logger.info("="*60)

        # Initialize and train model
        bert_detector = BERTPhishingDetector(max_length=512, random_state=self.random_state)
        bert_detector.train(X_train, X_test, y_train, y_test)

        # Evaluate model
        metrics = bert_detector.predict(X_test, y_test)

        # Save model
        model_path = os.path.join(PATHS['MODELS_DIR'], 'bert_model.pth')
        bert_detector.save_model(model_path)

        logger.info("\n" + "-"*50)
        logger.info("BERT MODEL RESULTS")
        logger.info("-"*50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"\nTraining Time:    {metrics['training_time']:.2f} seconds ({metrics['training_time'] / 60:.2f} minutes)")
        logger.info(f"Inference Time:   {metrics['inference_time']:.2f} seconds")
        logger.info("-"*50)

        self.models['bert'] = bert_detector
        return metrics

    def train_distilbert_model(self, X_train, X_test, y_train, y_test):
        """
        Train DistilBERT model.

        Args:
            X_train: Training texts
            X_test: Test texts
            y_train: Training labels
            y_test: Test labels

        Returns:
            dict: Model results
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2D: TRAINING DISTILBERT MODEL")
        logger.info("="*60)

        # Initialize and train model
        distilbert_detector = DistilBERTPhishingDetector(max_length=512, random_state=self.random_state)
        distilbert_detector.train(X_train, X_test, y_train, y_test)

        # Evaluate model
        metrics = distilbert_detector.predict(X_test, y_test)

        # Save model
        model_path = os.path.join(PATHS['MODELS_DIR'], 'distilbert_model.pth')
        distilbert_detector.save_model(model_path)

        logger.info("\n" + "-"*50)
        logger.info("DISTILBERT MODEL RESULTS")
        logger.info("-"*50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"\nTraining Time:    {metrics['training_time']:.2f} seconds ({metrics['training_time'] / 60:.2f} minutes)")
        logger.info(f"Inference Time:   {metrics['inference_time']:.2f} seconds")
        logger.info("-"*50)

        self.models['distilbert'] = distilbert_detector
        return metrics

    def run_pipeline(self, train_transformers=True, population_size=20, max_iter=50):
        """
        Run complete pipeline.

        Args:
            train_transformers: Whether to train transformer models (requires GPU)
            population_size: Population size for Dandelion optimization
            max_iter: Maximum iterations for Dandelion optimization

        Returns:
            dict: All results
        """
        start_time = time.time()

        # Phase 1: Data loading and preprocessing
        (X_train_ml, X_test_ml, y_train_ml, y_test_ml,
         X_train_tf, X_test_tf, y_train_tf, y_test_tf) = self.load_and_preprocess_data()

        # Phase 2: Train models
        self.results['naive_bayes'] = self.train_naive_bayes_model(
            X_train_ml, X_test_ml, y_train_ml, y_test_ml
        )

        self.results['dandelion_nb'] = self.train_dandelion_nb_model(
            X_train_ml, X_test_ml, y_train_ml, y_test_ml,
            population_size=population_size, max_iter=max_iter
        )

        if train_transformers:
            self.results['bert'] = self.train_bert_model(
                X_train_tf, X_test_tf, y_train_tf, y_test_tf
            )

            self.results['distilbert'] = self.train_distilbert_model(
                X_train_tf, X_test_tf, y_train_tf, y_test_tf
            )

        # Phase 3: Compare results
        self.compare_and_visualize_results()

        self.total_time = time.time() - start_time

        # Phase 4: Save results
        self.save_results()

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
            'dandelion_nb': 'NB + Dandelion',
            'bert': 'BERT',
            'distilbert': 'DistilBERT'
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
        plot_model_comparison(comparison_df, save_path=os.path.join(PATHS['PLOTS_DIR'], 'model_comparison.png'))

        # Training time comparison
        plot_training_time_comparison(comparison_df, save_path=os.path.join(PATHS['PLOTS_DIR'], 'training_time_comparison.png'))

        # Radar chart for metrics
        plot_model_comparison_radar(
            comparison_data,
            save_path=os.path.join(PATHS['PLOTS_DIR'], 'model_radar_chart.png')
        )

        # Accuracy vs. efficiency
        plot_accuracy_vs_efficiency(
            comparison_data,
            save_path=os.path.join(PATHS['PLOTS_DIR'], 'accuracy_vs_efficiency.png')
        )

        logger.info(f"Visualizations saved to {PATHS['PLOTS_DIR']}")

    def save_results(self):
        """Save results to multiple formats (TXT, JSON, CSV)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: SAVING RESULTS")
        logger.info("="*60)

        # Export to all formats
        export_results_to_text(self.results, os.path.join(PATHS['RESULTS_DIR'], 'results_summary.txt'))
        export_results_to_json(self.results, os.path.join(PATHS['RESULTS_DIR'], 'model_results.json'))
        export_results_to_csv(self.results, os.path.join(PATHS['RESULTS_DIR'], 'model_results.csv'))
        self.save_run_metadata()

        logger.info(f"\nAll results saved to {PATHS['RESULTS_DIR']}/")
        logger.info("  - results_summary.txt (human-readable)")
        logger.info("  - model_results.json (machine-readable)")
        logger.info("  - model_results.csv (spreadsheet-compatible)")

    def save_run_metadata(self):
        """Save run configuration and hardware metadata."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        except Exception:
            cuda_available = False
            device = "unknown"
            gpu_name = None

        metadata = {
            "paper": "Optimizing Phishing Detection: Comparative Analysis of Lightweight Machine Learning and Transformer Models",
            "random_state": self.random_state,
            "max_features": self.max_features,
            "splits": {
                "ml_train_test": "70/30 stratified",
                "transformer_train_test": "80/20 stratified",
                "transformer_train_validation": "80/20 stratified from transformer training split",
            },
            "models": {
                "naive_bayes": {"alpha": 1.0, "fit_prior": True},
                "dandelion_nb": {"population_size": 20, "max_iter": 50},
                "bert": {"model_name": "bert-base-uncased", "max_length": 512, "learning_rate": 2e-5, "batch_size": 16, "eval_batch_size": 64, "epochs": 10},
                "distilbert": {"model_name": "distilbert-base-uncased", "max_length": 512, "learning_rate": 3e-5, "batch_size": 16, "eval_batch_size": 64, "epochs": 10},
            },
            "hardware": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "device": device,
                "cuda_available": cuda_available,
                "gpu_name": gpu_name,
            },
            "total_time_seconds": self.total_time,
        }

        output_path = os.path.join(PATHS['RESULTS_DIR'], 'run_metadata.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(metadata, file, indent=4)
        logger.info(f"Run metadata saved to {output_path}")


def main():
    """Main function to run the pipeline."""
    logger.info("="*60)
    logger.info("PHISHING EMAIL DETECTION PIPELINE")
    logger.info("Based on: Optimizing Phishing Detection: Comparative")
    logger.info("Analysis of Lightweight Machine Learning and Transformer Models")
    logger.info("="*60)

    # Initialize pipeline
    pipeline = PhishingDetectionPipeline(max_features=5000, random_state=42)

    # Run pipeline
    results = pipeline.run_pipeline(train_transformers=True, population_size=20, max_iter=50)

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    model_names = ['naive_bayes', 'dandelion_nb', 'bert', 'distilbert']
    model_display_names = ['Naive Bayes', 'NB + Dandelion', 'BERT', 'DistilBERT']

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
