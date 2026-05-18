"""H3 experiment: calibrated fusion of text and URL/metadata scores."""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from phishing_detection.experiments.datasets import build_dataset_metadata, load_meajor_dataframe, split_by_source, split_random
from phishing_detection.experiments.features import build_text_preprocessor, build_url_meta_preprocessor, get_model_scores
from phishing_detection.experiments.metrics import best_threshold_for_recall, calculate_research_metrics, metrics_at_threshold
from phishing_detection.experiments.results import save_experiment_run, save_summary_table


logger = logging.getLogger(__name__)


def run_h3_fusion(
    sample_size: int | None = None,
    split: str = "random",
    force_download: bool = False,
    random_state: int = 42,
) -> dict:
    """Run H3 score fusion and threshold optimization."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("H3 EXPERIMENT: FUSION AND CALIBRATION")
    logger.info("=" * 60)

    df = load_meajor_dataframe(sample_size, random_state, force_download)
    metadata = build_dataset_metadata(df)
    metadata.update({"split_policy": split, "random_state": random_state, "hypothesis": "H3 fusion/calibration reduces false positives"})
    train_full_df, test_df = split_random(df, random_state=random_state) if split == "random" else split_by_source(df, random_state=random_state)
    train_df, val_df = train_test_split(
        train_full_df,
        test_size=0.2,
        random_state=random_state,
        stratify=train_full_df["label"],
    )

    text_model = CalibratedClassifierCV(
        Pipeline([
            ("tfidf", build_text_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        method="sigmoid",
        cv=3,
    )
    meta_model = CalibratedClassifierCV(
        Pipeline([
            ("features", build_url_meta_preprocessor(train_df)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        method="sigmoid",
        cv=3,
    )

    train_start = time.time()
    text_model.fit(train_df["text"], train_df["label"])
    meta_model.fit(train_df, train_df["label"])
    training_time = time.time() - train_start

    text_train_scores = get_model_scores(text_model, train_df["text"])
    meta_train_scores = get_model_scores(meta_model, train_df)
    text_val_scores = get_model_scores(text_model, val_df["text"])
    meta_val_scores = get_model_scores(meta_model, val_df)
    text_test_scores = get_model_scores(text_model, test_df["text"])
    meta_test_scores = get_model_scores(meta_model, test_df)

    fusion_train = np.column_stack([text_train_scores, meta_train_scores])
    fusion_val = np.column_stack([text_val_scores, meta_val_scores])
    fusion_test = np.column_stack([text_test_scores, meta_test_scores])
    stacking = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    stacking.fit(fusion_train, train_df["label"])

    results = []
    score_sets = {
        "text_only_calibrated": (text_val_scores, text_test_scores),
        "url_meta_only_calibrated": (meta_val_scores, meta_test_scores),
        "weighted_average_50_50": ((text_val_scores + meta_val_scores) / 2, (text_test_scores + meta_test_scores) / 2),
        "stacking_logreg": (stacking.predict_proba(fusion_val)[:, 1], stacking.predict_proba(fusion_test)[:, 1]),
    }

    for name, (val_scores, test_scores) in score_sets.items():
        default_pred = (test_scores >= 0.5).astype(int)
        metrics = calculate_research_metrics(test_df["label"], default_pred, test_scores)
        threshold_info = best_threshold_for_recall(val_df["label"], val_scores, 0.98)
        metrics.update({
            "model": name,
            "training_time": training_time,
            "inference_time": 0.0,
            "train_samples": int(len(train_df)),
            "validation_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "optimized_threshold": threshold_info["threshold"],
            "optimized_threshold_fpr": threshold_info["false_positive_rate"],
        })
        if threshold_info["threshold"] is not None:
            metrics.update(metrics_at_threshold(test_df["label"], test_scores, threshold_info["threshold"], "optimized_"))
        results.append(metrics)
        logger.info("%s: F1=%.4f, FPR=%.4f, opt_fpr=%s", name, metrics["f1_score"], metrics["false_positive_rate"], threshold_info["false_positive_rate"])

    metadata["total_time_seconds"] = time.time() - start_time
    outputs = save_experiment_run(
        experiment_name="h3_fusion",
        results=results,
        metadata=metadata,
        command=build_h3_command(sample_size, split, force_download, random_state),
        summary_title="H3 Fusion and Calibration",
    )
    outputs["summary_csv"] = save_summary_table("h3_fusion", build_h3_summary(results), "h3_fusion_summary.csv")
    return {"results": results, "metadata": metadata, "output_paths": outputs}


def build_h3_summary(results):
    rows = []
    for row in results:
        rows.append({
            "model": row["model"],
            "default_f1": row["f1_score"],
            "default_fpr": row["false_positive_rate"],
            "optimized_threshold": row.get("optimized_threshold"),
            "optimized_f1": row.get("optimized_f1_score"),
            "optimized_fpr": row.get("optimized_false_positive_rate"),
            "optimized_recall": row.get("optimized_recall"),
        })
    return rows


def build_h3_command(sample_size, split, force_download, random_state) -> str:
    parts = ["python3 run.py experiment h3-fusion", f"--split {split}", f"--random-state {random_state}"]
    if sample_size:
        parts.append(f"--sample-size {sample_size}")
    if force_download:
        parts.append("--force-download")
    return " ".join(parts)
