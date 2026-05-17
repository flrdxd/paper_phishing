"""
Evaluation Utilities for Phishing Email Detection

This module provides functions for calculating metrics, generating confusion matrices,
computing confidence intervals, and comparing model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    return metrics


def calculate_confidence_interval(scores, confidence=0.95):
    """
    Calculate confidence interval for metric scores.

    Args:
        scores: List of metric scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        dict: Dictionary with mean, std, and confidence interval
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Calculate margin of error using t-distribution
    from scipy import stats
    dof = len(scores) - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, dof)
    margin_of_error = t_critical * (std_score / np.sqrt(len(scores)))

    ci_lower = mean_score - margin_of_error
    ci_upper = mean_score + margin_of_error

    return {
        'mean': mean_score,
        'std': std_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_range': (ci_lower, ci_upper),
        'margin_of_error': margin_of_error
    }


def generate_confusion_matrix(y_true, y_pred, class_names=['Legitimate', 'Phishing']):
    """
    Generate confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        tuple: Confusion matrix and DataFrame
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return cm, cm_df


def plot_confusion_matrix(cm_df, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm_df: Confusion matrix DataFrame
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curve(y_true, y_scores, model_name, save_path=None):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")

    plt.close()

    return fpr, tpr, roc_auc


def plot_precision_recall_curve(y_true, y_scores, model_name, save_path=None):
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-recall curve saved to {save_path}")

    plt.close()

    return precision, recall, pr_auc


def compare_models_metrics(model_results):
    """
    Compare metrics across multiple models.

    Args:
        model_results: Dictionary of model results
            {
                'model_name': {
                    'accuracy': 0.xx,
                    'precision': 0.xx,
                    'recall': 0.xx,
                    'f1_score': 0.xx,
                    'training_time': xx.x,
                    'inference_time': xx.x
                }
            }

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []

    for model_name, results in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'Training Time (s)': results.get('training_time', 0),
            'Inference Time (s)': results.get('inference_time', 0)
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    """
    Plot model comparison as bar chart.

    Args:
        comparison_df: DataFrame with model comparison
        save_path: Path to save the plot (optional)
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(comparison_df['Model']))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, comparison_df[metric], width, label=metric)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")

    plt.close()


def plot_training_time_comparison(comparison_df, save_path=None):
    """
    Plot training time comparison.

    Args:
        comparison_df: DataFrame with model comparison
        save_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(comparison_df['Model'], comparison_df['Training Time (s)'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training time comparison plot saved to {save_path}")

    plt.close()


def format_results_table(comparison_df):
    """
    Format results table for display.

    Args:
        comparison_df: DataFrame with model comparison

    Returns:
        str: Formatted table as string
    """
    comparison_df_display = comparison_df.copy()

    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        comparison_df_display[col] = comparison_df_display[col].apply(lambda x: f"{x:.2%}")

    # Format times
    for col in ['Training Time (s)', 'Inference Time (s)']:
        comparison_df_display[col] = comparison_df_display[col].apply(lambda x: f"{x:.2f}")

    return comparison_df_display.to_string(index=False)


def calculate_improvement(baseline_metrics, optimized_metrics):
    """
    Calculate percentage improvement from baseline to optimized model.

    Args:
        baseline_metrics: Baseline model metrics
        optimized_metrics: Optimized model metrics

    Returns:
        dict: Percentage improvements
    """
    improvements = {}

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if baseline_metrics.get(metric) and optimized_metrics.get(metric):
            baseline_value = baseline_metrics[metric]
            optimized_value = optimized_metrics[metric]
            improvement = ((optimized_value - baseline_value) / baseline_value) * 100
            improvements[metric] = improvement

    return improvements


def generate_classification_report_text(y_true, y_pred, class_names=['Legitimate', 'Phishing']):
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        str: Classification report
    """
    return classification_report(y_true, y_pred, target_names=class_names)


def main():
    """Main function to test evaluation utilities."""
    logger.info("Testing evaluation utilities...")

    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)

    # Test metrics calculation
    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f"Metrics: {metrics}")

    # Test confidence interval
    scores = [0.95, 0.96, 0.94, 0.97, 0.95]
    ci = calculate_confidence_interval(scores)
    logger.info(f"Confidence Interval: {ci}")

    # Test confusion matrix
    cm, cm_df = generate_confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm_df}")

    # Test model comparison
    model_results = {
        'Naive Bayes': {
            'accuracy': 0.96,
            'precision': 0.98,
            'recall': 0.94,
            'f1_score': 0.96,
            'training_time': 10.5,
            'inference_time': 0.1
        },
        'Dandelion NB': {
            'accuracy': 0.99,
            'precision': 0.99,
            'recall': 0.99,
            'f1_score': 0.99,
            'training_time': 15.2,
            'inference_time': 0.15
        }
    }

    comparison_df = compare_models_metrics(model_results)
    logger.info(f"\nModel Comparison:\n{format_results_table(comparison_df)}")

    logger.info("Evaluation utilities test complete!")


if __name__ == "__main__":
    main()
