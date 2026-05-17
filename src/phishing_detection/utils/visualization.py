"""
Visualization Utilities for Phishing Email Detection

This module provides functions for generating word clouds, learning curves,
performance visualizations, and comparative analysis plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.colors import ListedColormap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def generate_word_cloud(text, title="Word Cloud", save_path=None, max_words=100, colormap='viridis'):
    """
    Generate word cloud from text.

    Args:
        text: Text string
        title: Title for the word cloud
        save_path: Path to save the word cloud (optional)
        max_words: Maximum number of words to display
        colormap: Color map for the word cloud

    Returns:
        WordCloud: WordCloud object
    """
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        max_words=max_words,
        colormap=colormap,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Word cloud saved to {save_path}")

    plt.show()
    plt.close()

    return wordcloud


def generate_phishing_word_cloud(phishing_text, save_path=None):
    """
    Generate word cloud for phishing emails.

    Args:
        phishing_text: Combined text of phishing emails
        save_path: Path to save the word cloud (optional)

    Returns:
        WordCloud: WordCloud object
    """
    return generate_word_cloud(
        phishing_text,
        title="Phishing Emails - Most Common Words",
        save_path=save_path,
        max_words=100,
        colormap='Reds'
    )


def generate_legitimate_word_cloud(legitimate_text, save_path=None):
    """
    Generate word cloud for legitimate emails.

    Args:
        legitimate_text: Combined text of legitimate emails
        save_path: Path to save the word cloud (optional)

    Returns:
        WordCloud: WordCloud object
    """
    return generate_word_cloud(
        legitimate_text,
        title="Legitimate Emails - Most Common Words",
        save_path=save_path,
        max_words=100,
        colormap='Blues'
    )


def generate_comparative_word_clouds(phishing_text, legitimate_text, save_path=None):
    """
    Generate side-by-side word clouds for comparison.

    Args:
        phishing_text: Combined text of phishing emails
        legitimate_text: Combined text of legitimate emails
        save_path: Path to save the comparison (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Phishing word cloud
    wordcloud_phishing = WordCloud(
        width=800, height=600,
        background_color='white',
        max_words=100,
        colormap='Reds',
        relative_scaling=0.5,
        min_font_size=10
    ).generate(phishing_text)

    ax1.imshow(wordcloud_phishing, interpolation='bilinear')
    ax1.set_title('Phishing Emails', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # Legitimate word cloud
    wordcloud_legitimate = WordCloud(
        width=800, height=600,
        background_color='white',
        max_words=100,
        colormap='Blues',
        relative_scaling=0.5,
        min_font_size=10
    ).generate(legitimate_text)

    ax2.imshow(wordcloud_legitimate, interpolation='bilinear')
    ax2.set_title('Legitimate Emails', fontsize=16, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparative word clouds saved to {save_path}")

    plt.show()
    plt.close()


def plot_learning_curve(train_scores, val_scores, title="Learning Curve", save_path=None):
    """
    Plot learning curve showing training and validation performance over epochs.

    Args:
        train_scores: List of training scores
        val_scores: List of validation scores
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-o', label='Training Score', linewidth=2, markersize=6)
    plt.plot(epochs, val_scores, 'r-o', label='Validation Score', linewidth=2, markersize=6)

    # Mark convergence point
    convergence_epoch = np.argmin(val_scores)
    if convergence_epoch < len(epochs):
        plt.axvline(x=epochs[convergence_epoch], color='g', linestyle='--',
                   label=f'Convergence at Epoch {epochs[convergence_epoch]}', alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curve saved to {save_path}")

    plt.show()
    plt.close()


def plot_training_losses(train_losses, val_losses, title="Training and Validation Loss", save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=6)

    # Mark convergence point
    convergence_epoch = np.argmin(val_losses)
    if convergence_epoch < len(epochs):
        plt.axvline(x=epochs[convergence_epoch], color='g', linestyle='--',
                   label=f'Convergence at Epoch {epochs[convergence_epoch]}', alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training loss curve saved to {save_path}")

    plt.show()
    plt.close()


def plot_model_comparison_radar(model_metrics, save_path=None):
    """
    Plot radar chart comparing model performance.

    Args:
        model_metrics: Dictionary of model metrics
            {
                'model_name': {'accuracy': 0.xx, 'precision': 0.xx, 'recall': 0.xx, 'f1_score': 0.xx}
            }
        save_path: Path to save the plot (optional)
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(categories)

    # Calculate angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Set y-axis limits
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_yticklabels([f'{x:.2f}' for x in [0.8, 0.85, 0.9, 0.95, 1.0]], fontsize=10)

    # Add title and legend
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Radar chart saved to {save_path}")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_accuracy_vs_efficiency(model_metrics, save_path=None):
    """
    Plot accuracy vs. computational efficiency.

    Args:
        model_metrics: Dictionary of model metrics
            {
                'model_name': {'accuracy': 0.xx, 'training_time': xx.x, 'inference_time': xx.x}
            }
        save_path: Path to save the plot (optional)
    """
    model_names = list(model_metrics.keys())
    accuracies = [model_metrics[name].get('accuracy', 0) for name in model_names]
    training_times = [model_metrics[name].get('training_time', 0) / 60 for name in model_names]  # Convert to minutes

    # Normalize for plotting
    max_accuracy = max(accuracies)
    max_time = max(training_times)

    norm_accuracies = [acc / max_accuracy for acc in accuracies]
    norm_times = [1 - (t / max_time) for t in training_times]  # Higher is better (faster)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    for i, (name, acc, time) in enumerate(zip(model_names, norm_accuracies, norm_times)):
        ax.scatter(acc, time, s=200, alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(name, (acc, time), fontsize=10, ha='center', va='center', fontweight='bold')

    # Add quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5)

    # Label quadrants
    ax.text(0.97, 0.9, 'High Accuracy,\nHigh Efficiency', fontsize=10, ha='right', va='top', style='italic', color='green')
    ax.text(0.93, 0.9, 'Low Accuracy,\nHigh Efficiency', fontsize=10, ha='left', va='top', style='italic', color='orange')
    ax.text(0.97, 0.1, 'High Accuracy,\nLow Efficiency', fontsize=10, ha='right', va='bottom', style='italic', color='red')
    ax.text(0.93, 0.1, 'Low Accuracy,\nLow Efficiency', fontsize=10, ha='left', va='bottom', style='italic', color='gray')

    ax.set_xlabel('Normalized Accuracy', fontsize=12)
    ax.set_ylabel('Normalized Efficiency (Faster is Better)', fontsize=12)
    ax.set_title('Accuracy vs. Computational Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Accuracy vs. efficiency plot saved to {save_path}")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_feature_importance(feature_names, importance_scores, top_n=20, save_path=None):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        top_n: Number of top features to display
        save_path: Path to save the plot (optional)
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = [importance_scores[i] for i in indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_scores, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Most important at top
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_dataset_distribution(df, save_path=None):
    """
    Plot dataset label distribution.

    Args:
        df: DataFrame with 'label' column
        save_path: Path to save the plot (optional)
    """
    label_counts = df['label'].value_counts()
    label_names = ['Legitimate', 'Phishing']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    ax1.bar(label_names, label_counts.values, color=['blue', 'red'], alpha=0.7)
    ax1.set_xlabel('Email Type', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Email Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, count in enumerate(label_counts.values):
        ax1.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Pie chart
    ax2.pie(label_counts.values, labels=label_names, autopct='%1.1f%%',
            colors=['blue', 'red'], startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Email Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dataset distribution plot saved to {save_path}")

    plt.show()
    plt.close()


def main():
    """Main function to test visualization utilities."""
    logger.info("Testing visualization utilities...")

    # Test word cloud with sample text
    sample_text = "phishing email security account verify click link urgent bank login password update"
    generate_word_cloud(sample_text, title="Test Word Cloud")

    # Test learning curve
    train_scores = [0.85, 0.90, 0.93, 0.95, 0.96, 0.965]
    val_scores = [0.83, 0.88, 0.91, 0.92, 0.925, 0.925]
    plot_learning_curve(train_scores, val_scores, title="Test Learning Curve")

    # Test model comparison radar
    model_metrics = {
        'Naive Bayes': {'accuracy': 0.96, 'precision': 0.98, 'recall': 0.94, 'f1_score': 0.96},
        'Dandelion NB': {'accuracy': 0.99, 'precision': 0.99, 'recall': 0.99, 'f1_score': 0.99}
    }
    plot_model_comparison_radar(model_metrics)

    logger.info("Visualization utilities test complete!")


if __name__ == "__main__":
    main()
