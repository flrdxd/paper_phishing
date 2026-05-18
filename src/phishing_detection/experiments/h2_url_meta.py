"""H2 experiment: compare text, URL/metadata, and combined features."""

from __future__ import annotations

import logging
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing_detection.experiments.datasets import (
    build_dataset_metadata,
    iter_leave_one_source_out,
    load_meajor_dataframe,
    split_by_source,
    split_random,
)
from phishing_detection.experiments.features import (
    build_text_preprocessor,
    build_text_url_meta_preprocessor,
    build_url_meta_preprocessor,
    get_model_scores,
)
from phishing_detection.experiments.metrics import calculate_research_metrics
from phishing_detection.experiments.results import save_experiment_run, save_summary_table


logger = logging.getLogger(__name__)


def run_h2_url_meta(
    sample_size: int | None = None,
    split: str = "random",
    force_download: bool = False,
    random_state: int = 42,
) -> dict:
    """Run H2 feature ablation on MeAJOR."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("H2 EXPERIMENT: URL/METADATA FEATURE ABLATION")
    logger.info("=" * 60)

    df = load_meajor_dataframe(sample_size, random_state, force_download)
    metadata = build_dataset_metadata(df)
    metadata.update({
        "split_policy": split,
        "random_state": random_state,
        "hypothesis": "H2 URL/metadados improve beyond text-only features",
    })

    results = []
    for split_name, train_df, test_df in build_h2_splits(df, split, random_state):
        configs = {
            "text_only": build_text_preprocessor(),
            "url_meta_only": build_url_meta_preprocessor(train_df),
            "text_url_meta": build_text_url_meta_preprocessor(train_df),
        }

        for name, preprocessor in configs.items():
            logger.info("Training H2 feature set: %s/%s", split_name, name)
            result = train_h2_feature_set(name, preprocessor, train_df, test_df, random_state)
            result["split_name"] = split_name
            results.append(result)
            logger.info("%s/%s: F1=%.4f, FPR=%.4f", split_name, name, result["f1_score"], result["false_positive_rate"])

    metadata["total_time_seconds"] = time.time() - start_time
    outputs = save_experiment_run(
        experiment_name="h2_url_meta",
        results=results,
        metadata=metadata,
        command=build_h2_command(sample_size, split, force_download, random_state),
        summary_title="H2 URL/Metadata Feature Ablation",
    )
    outputs["summary_csv"] = save_summary_table("h2_url_meta", build_h2_summary(results), "h2_ablation_summary.csv")
    return {"results": results, "metadata": metadata, "output_paths": outputs}


def build_h2_splits(df, split, random_state):
    if split == "random":
        train_df, test_df = split_random(df, random_state=random_state)
        return [("random", train_df, test_df)]
    if split == "source":
        train_df, test_df = split_by_source(df, random_state=random_state)
        return [("source", train_df, test_df)]
    if split == "leave-one-source-out":
        return iter_leave_one_source_out(df)
    raise ValueError("split must be 'random', 'source', or 'leave-one-source-out'")


def train_h2_feature_set(name, preprocessor, train_df, test_df, random_state):
    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]
    )
    X_train = train_df["text"] if name == "text_only" else train_df
    X_test = test_df["text"] if name == "text_only" else test_df
    train_start = time.time()
    pipeline.fit(X_train, train_df["label"])
    training_time = time.time() - train_start
    infer_start = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time = time.time() - infer_start
    y_scores = get_model_scores(pipeline, X_test)
    metrics = calculate_research_metrics(test_df["label"], y_pred, y_scores)
    metrics.update({
        "model": name,
        "training_time": training_time,
        "inference_time": inference_time,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
    })
    return metrics


def build_h2_summary(results):
    frame = pd.DataFrame(results)
    rows = []
    for model, group in frame.groupby("model", dropna=False):
        rows.append({
            "model": model,
            "runs": int(len(group)),
            "f1_mean": float(group["f1_score"].mean()),
            "f1_std": float(group["f1_score"].std(ddof=0)),
            "recall_mean": float(group["recall"].mean()),
            "fpr_mean": float(group["false_positive_rate"].mean()),
            "mcc_mean": float(group["mcc"].mean()),
        })
    return rows


def build_h2_command(sample_size, split, force_download, random_state) -> str:
    parts = [
        "python3 run.py experiment h2-url-meta",
        f"--split {split}",
        f"--random-state {random_state}",
    ]
    if sample_size:
        parts.append(f"--sample-size {sample_size}")
    if force_download:
        parts.append("--force-download")
    return " ".join(parts)
