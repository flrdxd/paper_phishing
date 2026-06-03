"""Paper comparison utilities for baseline reproduction."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Paper results from Table IV in "Optimizing P hishing Detection:
# Comparative Analysis of Lightweight Machine Learning and Transformer Models"
# IEEE WF-PST 2025
PAPER_RESULTS = {
    "naive_bayes": {
        "accuracy": 0.9618,
        "precision": 0.9776,
        "recall": 0.9482,
        "f1_score": 0.9627,
        "model": "Naive Bayes",
    },
    "dandelion_nb": {
        "accuracy": 0.9868,
        "precision": 0.9886,
        "recall": 0.9839,
        "f1_score": 0.9863,
        "model": "NB + Dandelion",
    },
    "bert": {
        "accuracy": 0.9948,
        "precision": 0.9926,
        "recall": 0.9974,
        "f1_score": 0.9950,
        "model": "BERT",
    },
    "distilbert": {
        "accuracy": 0.9936,
        "precision": 0.9967,
        "recall": 0.9910,
        "f1_score": 0.9939,
        "model": "DistilBERT",
    },
}


def compare_with_paper(model_key: str, our_metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare our results with paper results.

    Args:
        model_key: Model identifier (naive_bayes, dandelion_nb, bert, distilbert)
        our_metrics: Our experimental results

    Returns:
        dict with comparison data
    """
    if model_key not in PAPER_RESULTS:
        logger.warning(f"No paper results available for {model_key}")
        return {"error": f"No paper results for {model_key}"}

    paper_metrics = PAPER_RESULTS[model_key]
    comparison = {
        "model": paper_metrics["model"],
        "paper_results": paper_metrics,
        "our_results": {
            "accuracy": float(our_metrics.get("accuracy", 0)),
            "precision": float(our_metrics.get("precision", 0)),
            "recall": float(our_metrics.get("recall", 0)),
            "f1_score": float(our_metrics.get("f1_score", 0)),
        },
        "differences": {},
        "reproduction_status": None,
    }

    # Calculate differences
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        paper_val = paper_metrics[metric]
        our_val = comparison["our_results"][metric]

        diff_abs = our_val - paper_val
        diff_pct = (diff_abs / paper_val) * 100 if paper_val > 0 else 0

        comparison["differences"][metric] = {
            "absolute": round(diff_abs, 4),
            "percentage": round(diff_pct, 2),
            "paper_value": paper_val,
            "our_value": our_val,
        }

    # Determine reproduction status based on F1 score difference
    f1_diff = comparison["differences"]["f1_score"]["absolute"]
    if abs(f1_diff) <= 0.001:  # Within 0.1%
        comparison["reproduction_status"] = "excellent"
    elif abs(f1_diff) <= 0.005:  # Within 0.5%
        comparison["reproduction_status"] = "good"
    elif abs(f1_diff) <= 0.01:  # Within 1%
        comparison["reproduction_status"] = "acceptable"
    else:
        comparison["reproduction_status"] = "poor"

    return comparison


def format_comparison_table(comparisons: dict[str, dict]) -> str:
    """Format paper comparison results as a readable table.

    Args:
        comparisons: Dictionary of model comparisons from compare_with_paper

    Returns:
        Formatted markdown table
    """
    lines = [
        "# Paper Reproduction Comparison",
        "",
        "## Model Performance vs Paper Results",
        "",
        "| Model | Metric | Paper | Ours | Diff | % Change | Status |",
        "|---|---|---|---|---|---|---|",
    ]

    status_map = {
        "excellent": "✅ Excellent",
        "good": "✅ Good",
        "acceptable": "⚠️ Acceptable",
        "poor": "❌ Poor",
    }

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
    }

    for model_key, comparison in comparisons.items():
        if "error" in comparison:
            continue

        model_name = comparison["model"]
        overall_status = comparison["reproduction_status"]

        # Header row for model
        lines.append(f"| **{model_name}** | | | | | | **{status_map.get(overall_status, '?')}** |")

        # Individual metrics
        for metric_key, metric_name in metric_names.items():
            if metric_key not in comparison["differences"]:
                continue

            diff_data = comparison["differences"][metric_key]
            paper_val = diff_data["paper_value"]
            our_val = diff_data["our_value"]
            diff_abs = diff_data["absolute"]
            diff_pct = diff_data["percentage"]

            diff_sign = "+" if diff_abs >= 0 else ""
            lines.append(
                f"| {model_name} | {metric_name} | {paper_val:.4f} | {our_val:.4f} | "
                f"{diff_sign}{diff_abs:.4f} | {diff_sign}{diff_pct:.2f}% | |"
            )

        lines.append("|")

    lines.append("")
    lines.append("## Reproduction Status Legend")
    lines.append("- ✅ **Excellent**: F1 difference ≤ 0.1%")
    lines.append("- ✅ **Good**: F1 difference ≤ 0.5%")
    lines.append("- ⚠️ **Acceptable**: F1 difference ≤ 1%")
    lines.append("- ❌ **Poor**: F1 difference > 1%")
    lines.append("")

    return "\n".join(lines)


def get_paper_summary() -> dict[str, Any]:
    """Get summary of paper baseline results for reference.

    Returns:
        dict with paper metadata and results
    """
    return {
        "paper_title": "Optimizing Phishing Detection: Comparative Analysis of Lightweight Machine Learning and Transformer Models",
        "venue": "IEEE World Forum on Public Safety Technology (WF-PST)",
        "year": 2025,
        "dataset": "naserabdullahalam/phishing-email-dataset",
        "reported_samples": {
            "phishing": 42891,
            "legitimate": 39595,
            "total": 82486,
        },
        "models_tested": list(PAPER_RESULTS.keys()),
        "results": PAPER_RESULTS,
    }