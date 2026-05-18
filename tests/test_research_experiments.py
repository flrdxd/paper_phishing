"""Tests for research experiment helpers."""

from __future__ import annotations

import pandas as pd

from phishing_detection.check_kaggle_setup import find_kaggle_credentials
from phishing_detection.experiments.datasets import (
    iter_leave_one_source_out,
    normalize_meajor_dataframe,
    split_by_source,
    split_random,
)
from phishing_detection.experiments.h1_meajor import parse_model_list
from phishing_detection.experiments.features import (
    available_url_meta_columns,
    build_text_url_meta_preprocessor,
    build_url_meta_preprocessor,
)
from phishing_detection.experiments.h5_robustness import build_rewrites, remove_urgent_words
from phishing_detection.experiments.metrics import (
    best_threshold_for_recall,
    calculate_research_metrics,
    simulate_base_rates,
)
from phishing_detection.experiments import results as results_module


def make_meajor_like_dataframe() -> pd.DataFrame:
    rows = []
    for source_index, source in enumerate(["trec5", "nazario", "spamassassin", "enron"]):
        for label in [0.0, 1.0]:
            for item in range(8):
                rows.append(
                    {
                        "subject": f"Subject {source} {label} {item}",
                        "body": f"Body text from {source} with label {label} and item {item}",
                        "source": source,
                        "label": label,
                        "url_count": float(source_index),
                        "url_length_max": float(10 + source_index),
                        "url_length_avg": float(5 + source_index),
                        "url_subdom_max": float(source_index),
                        "url_subdom_avg": float(source_index),
                        "attachment_count": float(item % 2),
                        "has_attachments": bool(item % 2),
                        "sender_domain": f"{source}.example",
                        "receiver_domain": "corp.example",
                        "content_types": "text/plain",
                        "language": "en",
                    }
                )
    return pd.DataFrame(rows)


def test_normalize_meajor_dataframe_builds_text_and_integer_labels():
    raw_df = make_meajor_like_dataframe()

    normalized = normalize_meajor_dataframe(raw_df)

    assert {"text", "label", "source"}.issubset(normalized.columns)
    assert normalized["label"].dtype == "int64"
    assert set(normalized["label"].unique()) == {0, 1}
    assert normalized["text"].str.contains("Subject").all()
    assert normalized["text"].str.contains("Body text").all()


def test_random_split_is_stratified():
    df = normalize_meajor_dataframe(make_meajor_like_dataframe())

    train_df, test_df = split_random(df, random_state=42)

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}


def test_source_split_has_no_source_leakage():
    df = normalize_meajor_dataframe(make_meajor_like_dataframe())

    train_df, test_df = split_by_source(df, random_state=42)

    train_sources = set(train_df["source"].unique())
    test_sources = set(test_df["source"].unique())
    assert train_sources.isdisjoint(test_sources)
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}


def test_leave_one_source_out_splits_have_no_source_leakage():
    df = normalize_meajor_dataframe(make_meajor_like_dataframe())

    splits = iter_leave_one_source_out(df)

    assert len(splits) == 4
    for held_out, train_df, test_df in splits:
        assert held_out not in set(train_df["source"].unique())
        assert set(test_df["source"].unique()) == {held_out}
        assert set(test_df["label"].unique()) == {0, 1}


def test_research_metrics_include_operational_rates():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1]
    y_scores = [0.1, 0.8, 0.2, 0.9, 0.4, 0.7]

    metrics = calculate_research_metrics(y_true, y_pred, y_scores)

    assert metrics["false_positive_rate"] == 1 / 3
    assert metrics["false_negative_rate"] == 1 / 3
    assert metrics["true_negative_rate"] == 2 / 3
    assert metrics["auc_pr"] is not None
    assert metrics["fpr_at_recall_98"] is not None


def test_best_threshold_and_base_rate_simulation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.7, 0.8, 0.9, 0.95]

    threshold = best_threshold_for_recall(y_true, y_scores, target_recall=0.98)
    simulations = simulate_base_rates(y_true, [0, 0, 1, 1, 1, 1], base_rates=[0.01])

    assert threshold["threshold"] is not None
    assert simulations[0]["emails"] == 10_000
    assert "expected_false_positives" in simulations[0]


def test_url_meta_feature_builders_fit():
    df = normalize_meajor_dataframe(make_meajor_like_dataframe())

    numeric, categorical = available_url_meta_columns(df)
    url_meta = build_url_meta_preprocessor(df)
    combined = build_text_url_meta_preprocessor(df)

    assert "url_count" in numeric
    assert "sender_domain" in categorical
    assert url_meta.fit_transform(df).shape[0] == len(df)
    assert combined.fit_transform(df).shape[0] == len(df)


def test_controlled_rewrites_are_deterministic():
    text = "URGENT verify your password at https://example.test now"
    rewritten = build_rewrites(pd.Series([text]))

    assert "URGENT" not in remove_urgent_words(text)
    assert rewritten["url_placeholder"].iloc[0].count("LINK") == 1
    assert "Please review" in rewritten["corporate_tone"].iloc[0]


def test_parse_model_list_validates_names():
    assert parse_model_list("nb, logreg") == ["nb", "logreg"]

    try:
        parse_model_list("nb,unknown")
    except ValueError as exc:
        assert "Unknown model" in str(exc)
    else:
        raise AssertionError("Unknown model should raise ValueError")


def test_find_kaggle_credentials_supports_config_dir(tmp_path, monkeypatch):
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    credential_file = config_dir / "kaggle.json"
    credential_file.write_text("{}")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    assert find_kaggle_credentials() == str(credential_file)


def test_save_experiment_run_creates_timestamped_outputs(tmp_path, monkeypatch):
    monkeypatch.setitem(
        results_module.PATHS,
        "RESULTS_DIR",
        str(tmp_path / "results"),
    )

    outputs = results_module.save_experiment_run(
        experiment_name="unit_test_experiment",
        results=[
            {
                "model": "nb",
                "accuracy": 0.9,
                "precision": 0.8,
                "recall": 0.7,
                "f1_score": 0.75,
                "mcc": 0.5,
                "false_positive_rate": 0.1,
                "false_negative_rate": 0.2,
                "auc_pr": 0.85,
            }
        ],
        metadata={
            "dataset": "unit-test",
            "split_policy": "random",
            "random_state": 42,
            "samples": 10,
        },
        command="python3 run.py experiment unit-test",
        summary_title="Unit Test Experiment",
    )

    for path in outputs.values():
        assert path

    assert pd.read_csv(outputs["metrics_csv"]).iloc[0]["model"] == "nb"
    assert pd.read_csv(outputs["aggregate_csv"])["experiment"].str.contains(
        "unit_test_experiment"
    ).any()
