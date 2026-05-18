"""H5 experiment: controlled rewrite robustness stress test."""

from __future__ import annotations

import logging
import re
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing_detection.experiments.datasets import build_dataset_metadata, load_meajor_dataframe, split_random
from phishing_detection.experiments.features import build_text_preprocessor, get_model_scores
from phishing_detection.experiments.metrics import calculate_research_metrics
from phishing_detection.experiments.results import save_experiment_run


logger = logging.getLogger(__name__)

URGENT_WORDS = [
    "urgent",
    "immediately",
    "verify",
    "account",
    "password",
    "login",
    "suspended",
    "security",
    "confirm",
]


def run_h5_robustness(
    sample_size: int | None = None,
    force_download: bool = False,
    random_state: int = 42,
) -> dict:
    """Run controlled text rewrite stress tests."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("H5 EXPERIMENT: CONTROLLED ROBUSTNESS STRESS TEST")
    logger.info("=" * 60)

    df = load_meajor_dataframe(sample_size, random_state, force_download)
    train_df, test_df = split_random(df, random_state=random_state)
    metadata = build_dataset_metadata(df)
    metadata.update({"split_policy": "random", "random_state": random_state, "hypothesis": "H5 controlled rewrites reduce text-only robustness"})

    pipeline = Pipeline([
        ("tfidf", build_text_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    train_start = time.time()
    pipeline.fit(train_df["text"], train_df["label"])
    training_time = time.time() - train_start

    results = []
    original_scores = get_model_scores(pipeline, test_df["text"])
    original_pred = pipeline.predict(test_df["text"])
    original_metrics = calculate_research_metrics(test_df["label"], original_pred, original_scores)
    original_metrics.update({"model": "original", "training_time": training_time, "inference_time": 0.0, "train_samples": len(train_df), "test_samples": len(test_df), "robustness_gap": 0.0})
    results.append(original_metrics)

    for rewrite_name, rewritten_text in build_rewrites(test_df["text"]).items():
        infer_start = time.time()
        y_pred = pipeline.predict(rewritten_text)
        inference_time = time.time() - infer_start
        y_scores = get_model_scores(pipeline, rewritten_text)
        metrics = calculate_research_metrics(test_df["label"], y_pred, y_scores)
        metrics.update({
            "model": rewrite_name,
            "training_time": training_time,
            "inference_time": inference_time,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "robustness_gap": original_metrics["f1_score"] - metrics["f1_score"],
        })
        results.append(metrics)
        logger.info("%s: F1=%.4f, gap=%.4f", rewrite_name, metrics["f1_score"], metrics["robustness_gap"])

    metadata["total_time_seconds"] = time.time() - start_time
    outputs = save_experiment_run(
        experiment_name="h5_robustness",
        results=results,
        metadata=metadata,
        command=build_h5_command(sample_size, force_download, random_state),
        summary_title="H5 Controlled Robustness Stress Test",
    )
    return {"results": results, "metadata": metadata, "output_paths": outputs}


def build_rewrites(texts: pd.Series) -> dict[str, pd.Series]:
    """Build deterministic defensive stress-test rewrites."""
    return {
        "remove_urgent_words": texts.apply(remove_urgent_words),
        "corporate_tone": texts.apply(lambda text: "Please review this business communication. " + str(text)),
        "url_placeholder": texts.apply(lambda text: re.sub(r"https?://\S+|<\|URL\|>|\[URL\]", "LINK", str(text))),
        "shortened_text": texts.apply(lambda text: " ".join(str(text).split()[:80])),
    }


def remove_urgent_words(text: str) -> str:
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, URGENT_WORDS)) + r")\b", re.IGNORECASE)
    return pattern.sub("", str(text))


def build_h5_command(sample_size, force_download, random_state) -> str:
    parts = ["python3 run.py experiment h5-robustness", f"--random-state {random_state}"]
    if sample_size:
        parts.append(f"--sample-size {sample_size}")
    if force_download:
        parts.append("--force-download")
    return " ".join(parts)
