"""H6 experiment: realistic base-rate simulation."""

from __future__ import annotations

import logging
import time

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing_detection.experiments.datasets import build_dataset_metadata, load_meajor_dataframe, split_by_source, split_random
from phishing_detection.experiments.features import build_text_preprocessor, build_text_url_meta_preprocessor, build_url_meta_preprocessor, get_model_scores
from phishing_detection.experiments.metrics import calculate_research_metrics, simulate_base_rates
from phishing_detection.experiments.results import save_experiment_run, save_summary_table


logger = logging.getLogger(__name__)


def run_h6_base_rates(
    sample_size: int | None = None,
    split: str = "random",
    force_download: bool = False,
    random_state: int = 42,
) -> dict:
    """Run operational base-rate simulation with a text classifier."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("H6 EXPERIMENT: REALISTIC BASE RATE SIMULATION")
    logger.info("=" * 60)

    df = load_meajor_dataframe(sample_size, random_state, force_download)
    train_df, test_df = split_random(df, random_state=random_state) if split == "random" else split_by_source(df, random_state=random_state)
    metadata = build_dataset_metadata(df)
    metadata.update({"split_policy": split, "random_state": random_state, "hypothesis": "H6 base rates change operational model choice"})

    results = []
    configs = {
        "text_only": (build_text_preprocessor(), "text"),
        "url_meta_only": (build_url_meta_preprocessor(train_df), "frame"),
        "text_url_meta": (build_text_url_meta_preprocessor(train_df), "frame"),
    }

    for model_name, (preprocessor, input_kind) in configs.items():
        pipeline = Pipeline([
            ("features", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ])
        X_train = train_df["text"] if input_kind == "text" else train_df
        X_test = test_df["text"] if input_kind == "text" else test_df
        train_start = time.time()
        pipeline.fit(X_train, train_df["label"])
        training_time = time.time() - train_start
        infer_start = time.time()
        y_pred = pipeline.predict(X_test)
        inference_time = time.time() - infer_start
        y_scores = get_model_scores(pipeline, X_test)
        metrics = calculate_research_metrics(test_df["label"], y_pred, y_scores)
        for simulation in simulate_base_rates(test_df["label"], y_pred):
            row = dict(metrics)
            row.update(simulation)
            row.update({
                "model": model_name,
                "training_time": training_time,
                "inference_time": inference_time,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "threshold_policy": "default_0.5",
            })
            results.append(row)

    metadata["total_time_seconds"] = time.time() - start_time
    outputs = save_experiment_run(
        experiment_name="h6_base_rates",
        results=results,
        metadata=metadata,
        command=build_h6_command(sample_size, split, force_download, random_state),
        summary_title="H6 Realistic Base Rate Simulation",
    )
    outputs["summary_csv"] = save_summary_table("h6_base_rates", build_h6_summary(results), "h6_operational_summary.csv")
    return {"results": results, "metadata": metadata, "output_paths": outputs}


def build_h6_summary(results):
    rows = []
    for row in results:
        rows.append({
            "model": row["model"],
            "base_rate": row["base_rate"],
            "expected_false_positives": row["expected_false_positives"],
            "expected_false_negatives": row["expected_false_negatives"],
            "flagged_percent": row["flagged_percent"],
            "fpr": row["false_positive_rate"],
            "fnr": row["false_negative_rate"],
        })
    return rows


def build_h6_command(sample_size, split, force_download, random_state) -> str:
    parts = ["python3 run.py experiment h6-base-rates", f"--split {split}", f"--random-state {random_state}"]
    if sample_size:
        parts.append(f"--sample-size {sample_size}")
    if force_download:
        parts.append("--force-download")
    return " ".join(parts)
