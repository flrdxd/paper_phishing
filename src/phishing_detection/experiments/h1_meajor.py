"""H1 experiment: evaluate lightweight models on the MeAJOR corpus."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from phishing_detection.experiments.datasets import (
    build_dataset_metadata,
    iter_leave_one_source_out,
    load_meajor_dataframe,
    split_by_source,
    split_random,
)
from phishing_detection.experiments.metrics import calculate_research_metrics
from phishing_detection.experiments.results import save_experiment_run, save_summary_table
from phishing_detection.path_config import PATHS


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "nb": lambda random_state: MultinomialNB(),
    "logreg": lambda random_state: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    ),
    "svm": lambda random_state: LinearSVC(
        class_weight="balanced",
        random_state=random_state,
    ),
}


def parse_model_list(models: str) -> list[str]:
    """Parse and validate a comma-separated model list."""
    selected = [model.strip().lower() for model in models.split(",") if model.strip()]
    unknown = sorted(set(selected) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Available models: {sorted(MODEL_REGISTRY)}"
        )
    return selected or ["nb", "logreg", "svm"]


def run_h1_meajor(
    sample_size: int | None = None,
    split: str = "random",
    models: str = "nb,logreg,svm",
    force_download: bool = False,
    random_state: int = 42,
) -> dict:
    """Run the H1 MeAJOR experiment and persist metrics/metadata."""
    start_time = time.time()
    selected_models = parse_model_list(models)

    logger.info("=" * 60)
    logger.info("H1 EXPERIMENT: MEAJOR MULTI-SOURCE DATASET")
    logger.info("=" * 60)
    logger.info("Models: %s", ", ".join(selected_models))
    logger.info("Split policy: %s", split)
    logger.info("Sample size: %s", sample_size or "full dataset")

    df = load_meajor_dataframe(
        sample_size=sample_size,
        random_state=random_state,
        force_download=force_download,
    )
    metadata = build_dataset_metadata(df)
    metadata["split_policy"] = split
    metadata["selected_models"] = selected_models
    metadata["random_state"] = random_state

    results = []
    for split_name, train_df, test_df in build_h1_splits(df, split, random_state):
        logger.info("Split %s: Train=%s (%s sources), Test=%s (%s sources)", split_name, len(train_df), train_df["source"].nunique(), len(test_df), test_df["source"].nunique())
        for model_name in selected_models:
            logger.info("-" * 60)
            logger.info("Training H1 model: %s", model_name)
            result = train_and_evaluate_model(
                model_name=model_name,
                train_df=train_df,
                test_df=test_df,
                random_state=random_state,
            )
            result["split_name"] = split_name
            result["train_sources"] = int(train_df["source"].nunique())
            result["test_sources"] = int(test_df["source"].nunique())
            results.append(result)
            logger.info("%s/%s: F1=%.4f, Recall=%.4f, FPR=%.4f, MCC=%.4f", split_name, model_name, result["f1_score"], result["recall"], result["false_positive_rate"], result["mcc"])

    metadata["total_time_seconds"] = float(time.time() - start_time)
    output_paths = save_h1_outputs(
        results=results,
        metadata=metadata,
        command=build_h1_command(sample_size, split, models, force_download, random_state),
    )
    output_paths["summary_csv"] = save_summary_table("h1_meajor", build_h1_summary(results), "h1_generalization_summary.csv")
    logger.info("H1 outputs saved: %s", output_paths)
    logger.info("=" * 60)
    logger.info("H1 experiment complete")
    logger.info("=" * 60)

    return {
        "results": results,
        "metadata": metadata,
        "output_paths": output_paths,
    }


def build_h1_splits(df: pd.DataFrame, split: str, random_state: int):
    if split == "random":
        train_df, test_df = split_random(df, random_state=random_state)
        return [("random", train_df, test_df)]
    if split == "source":
        train_df, test_df = split_by_source(df, random_state=random_state)
        return [("source", train_df, test_df)]
    if split == "leave-one-source-out":
        return iter_leave_one_source_out(df)
    raise ValueError("split must be 'random', 'source', or 'leave-one-source-out'")


def build_h1_summary(results: list[dict]) -> list[dict]:
    frame = pd.DataFrame(results)
    grouped = frame.groupby("model", dropna=False)
    rows = []
    for model, group in grouped:
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


def train_and_evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
) -> dict:
    """Train one lightweight text model and return research metrics."""
    estimator = MODEL_REGISTRY[model_name](random_state)
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    min_df=5,
                    max_df=0.8,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            ("model", estimator),
        ]
    )

    train_start = time.time()
    pipeline.fit(train_df["text"], train_df["label"])
    training_time = time.time() - train_start

    inference_start = time.time()
    y_pred = pipeline.predict(test_df["text"])
    inference_time = time.time() - inference_start
    y_scores = get_model_scores(pipeline, test_df["text"])

    metrics = calculate_research_metrics(test_df["label"], y_pred, y_scores)
    metrics.update(
        {
            "model": model_name,
            "training_time": float(training_time),
            "inference_time": float(inference_time),
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
        }
    )
    return metrics


def get_model_scores(pipeline: Pipeline, texts: pd.Series):
    """Return phishing scores from predict_proba or decision_function."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        return scores
    return None


def build_h1_command(
    sample_size: int | None,
    split: str,
    models: str,
    force_download: bool,
    random_state: int,
) -> str:
    """Build the reproducible command string for this H1 run."""
    parts = [
        "python3 run.py experiment h1-meajor",
        f"--split {split}",
        f"--models {models}",
        f"--random-state {random_state}",
    ]
    if sample_size:
        parts.append(f"--sample-size {sample_size}")
    if force_download:
        parts.append("--force-download")
    return " ".join(parts)


def save_h1_outputs(results: list[dict], metadata: dict, command: str) -> dict:
    """Save H1 metrics and metadata under research artifacts."""
    output_dir = Path(PATHS["RESULTS_DIR"]) / "research"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "h1_meajor_metrics.csv"
    metrics_json = output_dir / "h1_meajor_metrics.json"
    metadata_json = output_dir / "h1_meajor_metadata.json"

    pd.DataFrame(results).to_csv(metrics_csv, index=False)
    with metrics_json.open("w") as file:
        json.dump(results, file, indent=4)
    with metadata_json.open("w") as file:
        json.dump(metadata, file, indent=4)

    run_outputs = save_experiment_run(
        experiment_name="h1_meajor",
        results=results,
        metadata=metadata,
        command=command,
        summary_title="H1 MeAJOR Experiment",
    )

    return {
        "latest_metrics_csv": str(metrics_csv),
        "latest_metrics_json": str(metrics_json),
        "latest_metadata_json": str(metadata_json),
        "run_dir": run_outputs["run_dir"],
        "run_metrics_csv": run_outputs["metrics_csv"],
        "run_metrics_json": run_outputs["metrics_json"],
        "run_metadata_json": run_outputs["metadata_json"],
        "command_txt": run_outputs["command_txt"],
        "summary_md": run_outputs["summary_md"],
        "aggregate_csv": run_outputs["aggregate_csv"],
    }
