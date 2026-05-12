"""
Results Exporter for Phishing Email Detection

This module provides functions to export results to various formats:
- Text file (human-readable)
- JSON file (machine-readable)
- CSV file (spreadsheet-compatible)
"""

import os
import json
import csv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_results_to_text(results, output_path='results/results_summary.txt'):
    """
    Export model results to a human-readable text file.

    Args:
        results: Dictionary of model results
        output_path: Path to save the text file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("PHISHING EMAIL DETECTION - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Based on paper: 'Optimizing Phishing Detection: Comparative Analysis\n")
        f.write("of Lightweight Machine Learning and Transformer Models'\n")
        f.write("IEEE World Forum on Public Safety Technology (WF-PST) 2025\n\n")

        # Results from paper for comparison
        f.write("-"*80 + "\n")
        f.write("EXPECTED RESULTS FROM PAPER (for comparison):\n")
        f.write("-"*80 + "\n")
        f.write("Naive Bayes:          Accuracy=96.18%, Precision=97.76%, Recall=94.82%, F1=96.27%\n")
        f.write("NB + Dandelion:       Accuracy=98.68%, Precision=98.86%, Recall=98.39%, F1=98.63%\n")
        f.write("BERT:                 Accuracy=99.48%, Precision=99.26%, Recall=99.74%, F1=99.50%\n")
        f.write("DistilBERT:           Accuracy=99.36%, Precision=99.67%, Recall=99.10%, F1=99.39%\n")
        f.write("\n")

        # Actual results
        f.write("-"*80 + "\n")
        f.write("ACTUAL RESULTS:\n")
        f.write("-"*80 + "\n\n")

        model_names = {
            'naive_bayes': 'Naive Bayes',
            'dandelion_nb': 'NB + Dandelion',
            'bert': 'BERT',
            'distilbert': 'DistilBERT'
        }

        for model_key, model_display_name in model_names.items():
            if model_key in results:
                metrics = results[model_key]
                f.write(f"{model_display_name}:\n")
                f.write(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision:      {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall:         {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score:       {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
                f.write(f"  Training Time:   {metrics['training_time']:.2f} seconds ({metrics['training_time']/60:.2f} minutes)\n")
                if 'inference_time' in metrics:
                    f.write(f"  Inference Time:  {metrics['inference_time']:.2f} seconds\n")
                if 'optimization_time' in metrics:
                    f.write(f"  Optimization Time: {metrics['optimization_time']:.2f} seconds\n")
                    f.write(f"  Total Time:         {metrics.get('total_time', 0):.2f} seconds\n")
                if 'alpha' in metrics:
                    f.write(f"  Optimized Parameters: alpha={metrics['alpha']:.4f}, fit_prior={metrics['fit_prior']}\n")
                f.write("\n")

        # Comparison table
        f.write("-"*80 + "\n")
        f.write("COMPARISON TABLE:\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time (s)':<12}\n")
        f.write("-"*80 + "\n")

        for model_key, model_display_name in model_names.items():
            if model_key in results:
                metrics = results[model_key]
                training_time = metrics.get('total_time', metrics['training_time'])
                f.write(f"{model_display_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                       f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {training_time:<12.2f}\n")

        f.write("\n")

        # Key findings
        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-"*80 + "\n\n")

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
        best_model_name = model_names.get(best_model[0], best_model[0])
        f.write(f"1. Best Performing Model: {best_model_name}\n")
        f.write(f"   F1-Score: {best_model[1]['f1_score']:.4f}\n\n")

        # Calculate improvements
        if 'naive_bayes' in results and 'dandelion_nb' in results:
            baseline = results['naive_bayes']
            optimized = results['dandelion_nb']
            f.write("2. Dandelion Optimization Impact:\n")
            improvements = {
                'Accuracy': ((optimized['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100,
                'Precision': ((optimized['precision'] - baseline['precision']) / baseline['precision']) * 100,
                'Recall': ((optimized['recall'] - baseline['recall']) / baseline['recall']) * 100,
                'F1-Score': ((optimized['f1_score'] - baseline['f1_score']) / baseline['f1_score']) * 100
            }
            for metric, improvement in improvements.items():
                f.write(f"   {metric}: +{improvement:.2f}%\n")
            f.write("\n")

        # Timing analysis
        f.write("3. Computational Efficiency:\n")
        for model_key, model_display_name in model_names.items():
            if model_key in results:
                metrics = results[model_key]
                training_time = metrics.get('total_time', metrics['training_time'])
                f.write(f"   {model_display_name}: {training_time:.2f}s ({training_time/60:.2f} min)\n")
        f.write("\n")

        # Recommendations
        f.write("-"*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n\n")
        f.write("- Use Naive Bayes for:\n")
        f.write("  * Fast, real-time detection\n")
        f.write("  * Resource-constrained environments\n")
        f.write("  * Systems without GPU\n\n")

        f.write("- Use Dandelion-optimized Naive Bayes for:\n")
        f.write("  * Balanced performance and efficiency\n")
        f.write("  * Improved accuracy over baseline\n")
        f.write("  * Moderate resource requirements\n\n")

        if 'bert' in results or 'distilbert' in results:
            f.write("- Use BERT/DistilBERT for:\n")
            f.write("  * Maximum accuracy requirements\n")
            f.write("  * Complex text pattern recognition\n")
            f.write("  * Systems with GPU resources\n\n")

        f.write("-"*80 + "\n")
        f.write("CONCLUSION:\n")
        f.write("-"*80 + "\n")
        f.write("All models successfully detect phishing emails with high accuracy.\n")
        f.write("Model selection should be based on the trade-off between\n")
        f.write("accuracy requirements and available computational resources.\n")
        f.write("="*80 + "\n")

    logger.info(f"Results exported to {output_path}")


def export_results_to_json(results, output_path='results/model_results.json'):
    """
    Export model results to JSON file.

    Args:
        results: Dictionary of model results
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare results for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            'accuracy': float(value.get('accuracy', 0)),
            'precision': float(value.get('precision', 0)),
            'recall': float(value.get('recall', 0)),
            'f1_score': float(value.get('f1_score', 0)),
            'training_time': float(value.get('training_time', 0)),
            'inference_time': float(value.get('inference_time', 0))
        }

        # Add optional fields
        if 'optimization_time' in value:
            serializable_results[key]['optimization_time'] = float(value['optimization_time'])
        if 'total_time' in value:
            serializable_results[key]['total_time'] = float(value['total_time'])
        if 'alpha' in value:
            serializable_results[key]['alpha'] = float(value['alpha'])
        if 'fit_prior' in value:
            serializable_results[key]['fit_prior'] = bool(value['fit_prior'])

    # Add metadata
    serializable_results['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'paper': 'Optimizing Phishing Detection: Comparative Analysis of Lightweight Machine Learning and Transformer Models',
        'conference': 'IEEE World Forum on Public Safety Technology (WF-PST) 2025'
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    logger.info(f"Results exported to {output_path}")


def export_results_to_csv(results, output_path='results/model_results.csv'):
    """
    Export model results to CSV file.

    Args:
        results: Dictionary of model results
        output_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_names = {
        'naive_bayes': 'Naive Bayes',
        'dandelion_nb': 'NB + Dandelion',
        'bert': 'BERT',
        'distilbert': 'DistilBERT'
    }

    # Prepare data for CSV
    csv_data = []
    for model_key, model_display_name in model_names.items():
        if model_key in results:
            metrics = results[model_key]
            row = {
                'Model': model_display_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'Training Time (s)': metrics.get('training_time', 0),
                'Inference Time (s)': metrics.get('inference_time', 0)
            }

            # Add optional fields
            if 'optimization_time' in metrics:
                row['Optimization Time (s)'] = metrics['optimization_time']
                row['Total Time (s)'] = metrics.get('total_time', 0)

            if 'alpha' in metrics:
                row['Alpha'] = metrics['alpha']
                row['Fit Prior'] = metrics['fit_prior']

            csv_data.append(row)

    # Write to CSV
    if csv_data:
        # Get all unique fieldnames from all rows to handle different models having different fields
        fieldnames = set()
        for row in csv_data:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(csv_data)

    logger.info(f"Results exported to {output_path}")


def export_all_formats(results):
    """
    Export results to all available formats.

    Args:
        results: Dictionary of model results
    """
    logger.info("Exporting results to all formats...")

    export_results_to_text(results)
    export_results_to_json(results)
    export_results_to_csv(results)

    logger.info("All results exported successfully!")


def main():
    """Test the results exporter."""
    # Sample results for testing
    sample_results = {
        'naive_bayes': {
            'accuracy': 0.9618,
            'precision': 0.9776,
            'recall': 0.9482,
            'f1_score': 0.9627,
            'training_time': 10.5,
            'inference_time': 0.1
        },
        'dandelion_nb': {
            'accuracy': 0.9868,
            'precision': 0.9869,
            'recall': 0.9867,
            'f1_score': 0.9873,
            'training_time': 8.2,
            'optimization_time': 120.5,
            'total_time': 128.7,
            'inference_time': 0.15,
            'alpha': 0.5234,
            'fit_prior': True
        }
    }

    export_all_formats(sample_results)


if __name__ == "__main__":
    main()
