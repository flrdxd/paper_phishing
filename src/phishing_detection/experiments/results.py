"""Shared result persistence for baseline and research experiments."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from phishing_detection.path_config import PATHS


def create_run_dir(experiment_name: str, timestamp: str | None = None) -> Path:
    """Create and return a timestamped run directory."""
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(PATHS["RESULTS_DIR"]) / "runs" / experiment_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_git_commit() -> str | None:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def get_hardware_metadata() -> dict[str, Any]:
    """Collect lightweight runtime and hardware metadata."""
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        metadata.update(
            {
                "device": "cuda" if cuda_available else "cpu",
                "cuda_available": cuda_available,
                "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            }
        )
    except Exception:
        metadata.update(
            {
                "device": "unknown",
                "cuda_available": False,
                "gpu_name": None,
            }
        )
    return metadata


def normalize_metric_rows(
    results: list[dict[str, Any]],
    experiment_name: str,
    dataset: str,
    split_policy: str,
    random_state: int,
) -> list[dict[str, Any]]:
    """Attach common experiment columns to metric rows."""
    rows = []
    for result in results:
        row = dict(result)
        row.setdefault("model", result.get("model", "unknown"))
        row["experiment"] = experiment_name
        row["dataset"] = dataset
        row["split_policy"] = split_policy
        row["random_state"] = random_state
        rows.append(row)
    return rows


def serialize_json_value(value: Any) -> Any:
    """Convert common non-JSON values into JSON-safe values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serialize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_json_value(item) for item in value]
    return value


def save_baseline_run(
    experiment_name: str,
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    command: str,
    summary_title: str,
    resource_logs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save baseline paper reproduction run WITHOUT adding to research aggregate table.

    CRITICAL: Baseline results MUST NOT be mixed with research experiments (H1-H6).
    This function saves only to the timestamped run directory, NOT to all_experiments_metrics.csv.

    Args:
        resource_logs: Complete resource monitoring data including CPU, RAM, GPU metrics
    """
    metadata = dict(metadata)
    metadata.setdefault("git_commit", get_git_commit())
    metadata.setdefault("hardware", get_hardware_metadata())
    metadata.setdefault("generated_at", datetime.now().isoformat())

    # Include resource monitoring in metadata
    if resource_logs:
        metadata["resource_monitoring"] = serialize_json_value(resource_logs)

    dataset = str(metadata.get("dataset", "unknown"))
    split_policy = str(metadata.get("split_policy", "unknown"))
    random_state = int(metadata.get("random_state", 42))
    metric_rows = normalize_metric_rows(
        results=results,
        experiment_name=experiment_name,
        dataset=dataset,
        split_policy=split_policy,
        random_state=random_state,
    )

    # Save to baseline/ directory instead of runs/baseline_paper/
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(PATHS["RESULTS_DIR"]) / "baseline" / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"
    command_txt = run_dir / "command.txt"
    summary_md = run_dir / "summary.md"
    resources_json = run_dir / "resource_monitoring.json"

    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    with metrics_json.open("w") as file:
        json.dump(serialize_json_value(metric_rows), file, indent=4)
    with metadata_json.open("w") as file:
        json.dump(serialize_json_value(metadata), file, indent=4)
    command_txt.write_text(command + "\n")
    summary_md.write_text(build_summary(summary_title, metric_rows, metadata))

    # Save complete resource monitoring data
    if resource_logs:
        with resources_json.open("w") as file:
            json.dump(serialize_json_value(resource_logs), file, indent=4)

    # NOTE: NO aggregate_csv append - baseline stays separate from research experiments
    return_dict = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "metadata_json": str(metadata_json),
        "command_txt": str(command_txt),
        "summary_md": str(summary_md),
    }

    if resource_logs:
        return_dict["resources_json"] = str(resources_json)

    return return_dict


def save_experiment_run(
    experiment_name: str,
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    command: str,
    summary_title: str,
    resource_logs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save research experiment run and update the aggregate metrics table.

    Use this function for H1-H6 research experiments only.
    For H0 baseline reproduction, use save_baseline_run() instead.

    Args:
        resource_logs: Complete resource monitoring data including CPU, RAM, GPU metrics
    """
    metadata = dict(metadata)
    metadata.setdefault("git_commit", get_git_commit())
    metadata.setdefault("hardware", get_hardware_metadata())
    metadata.setdefault("generated_at", datetime.now().isoformat())

    # Include resource monitoring in metadata
    if resource_logs:
        metadata["resource_monitoring"] = serialize_json_value(resource_logs)

    dataset = str(metadata.get("dataset", "unknown"))
    split_policy = str(metadata.get("split_policy", "unknown"))
    random_state = int(metadata.get("random_state", 42))
    metric_rows = normalize_metric_rows(
        results=results,
        experiment_name=experiment_name,
        dataset=dataset,
        split_policy=split_policy,
        random_state=random_state,
    )

    run_dir = create_run_dir(experiment_name)
    metrics_csv = run_dir / "metrics.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"
    command_txt = run_dir / "command.txt"
    summary_md = run_dir / "summary.md"
    resources_json = run_dir / "resource_monitoring.json"

    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    with metrics_json.open("w") as file:
        json.dump(serialize_json_value(metric_rows), file, indent=4)
    with metadata_json.open("w") as file:
        json.dump(serialize_json_value(metadata), file, indent=4)
    command_txt.write_text(command + "\n")
    summary_md.write_text(build_summary(summary_title, metric_rows, metadata))

    # Save complete resource monitoring data
    if resource_logs:
        with resources_json.open("w") as file:
            json.dump(serialize_json_value(resource_logs), file, indent=4)

    # Research experiments ARE added to aggregate table
    aggregate_csv = append_aggregate_metrics(metric_rows)

    # Build return dict - only include resources_json if resource_logs were provided
    return_dict = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "metadata_json": str(metadata_json),
        "command_txt": str(command_txt),
        "summary_md": str(summary_md),
        "aggregate_csv": str(aggregate_csv),
    }

    if resource_logs:
        return_dict["resources_json"] = str(resources_json)

    return return_dict


def save_summary_table(experiment_name: str, rows: list[dict[str, Any]], filename: str) -> str:
    """Save a clean per-experiment summary table under research summaries."""
    output_dir = Path(PATHS["RESULTS_DIR"]) / "research" / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    pd.DataFrame([serialize_json_value(row) for row in rows]).to_csv(output_path, index=False)
    return str(output_path)


def append_aggregate_metrics(rows: list[dict[str, Any]]) -> Path:
    """Append metric rows to the project-level aggregate research table."""
    output_path = Path(PATHS["RESULTS_DIR"]) / "research" / "all_experiments_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_rows = [serialize_json_value(row) for row in rows]
    if output_path.exists():
        existing_rows = pd.read_csv(output_path).to_dict(orient="records")
    else:
        existing_rows = []

    all_rows = existing_rows + serializable_rows
    fieldnames = sorted({key for row in all_rows for key in row})

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(all_rows)

    return output_path


def build_summary(
    title: str,
    metric_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Build a concise Markdown run summary."""
    lines = [
        f"# {title}",
        "",
        f"- Dataset: {metadata.get('dataset', 'unknown')}",
        f"- Split: {metadata.get('split_policy', 'unknown')}",
        f"- Samples: {metadata.get('samples', 'unknown')}",
        f"- Git commit: {metadata.get('git_commit', 'unknown')}",
        "",
        "## Metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | AUC-PR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{f1_score:.4f} | {mcc} | {fpr} | {fnr} | {auc_pr} |".format(
                model=row.get("model", "unknown"),
                accuracy=float(row.get("accuracy", 0) or 0),
                precision=float(row.get("precision", 0) or 0),
                recall=float(row.get("recall", 0) or 0),
                f1_score=float(row.get("f1_score", 0) or 0),
                mcc=_format_optional_float(row.get("mcc")),
                fpr=_format_optional_float(row.get("false_positive_rate")),
                fnr=_format_optional_float(row.get("false_negative_rate")),
                auc_pr=_format_optional_float(row.get("auc_pr")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def save_implementacao_run(
    experiment_name: str,
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    command: str,
    summary_title: str,
    resource_logs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save improved implementation run separate from baseline and research.

    Use this function for new implementations and improvements to the baseline code.
    Results are saved to implementacao/ directory to track code improvements over time.

    Args:
        resource_logs: Complete resource monitoring data including CPU, RAM, GPU metrics
    """
    metadata = dict(metadata)
    metadata.setdefault("git_commit", get_git_commit())
    metadata.setdefault("hardware", get_hardware_metadata())
    metadata.setdefault("generated_at", datetime.now().isoformat())

    # Include resource monitoring in metadata
    if resource_logs:
        metadata["resource_monitoring"] = serialize_json_value(resource_logs)

    dataset = str(metadata.get("dataset", "unknown"))
    split_policy = str(metadata.get("split_policy", "unknown"))
    random_state = int(metadata.get("random_state", 42))
    metric_rows = normalize_metric_rows(
        results=results,
        experiment_name=experiment_name,
        dataset=dataset,
        split_policy=split_policy,
        random_state=random_state,
    )

    # Save to implementacao/ directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(PATHS["RESULTS_DIR"]) / "implementacao" / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"
    command_txt = run_dir / "command.txt"
    summary_md = run_dir / "summary.md"
    resources_json = run_dir / "resource_monitoring.json"

    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    with metrics_json.open("w") as file:
        json.dump(serialize_json_value(metric_rows), file, indent=4)
    with metadata_json.open("w") as file:
        json.dump(serialize_json_value(metadata), file, indent=4)
    command_txt.write_text(command + "\n")
    summary_md.write_text(build_summary(summary_title, metric_rows, metadata))

    # Save complete resource monitoring data
    if resource_logs:
        with resources_json.open("w") as file:
            json.dump(serialize_json_value(resource_logs), file, indent=4)

    # NOTE: NO aggregate_csv append - implementacao stays separate
    return_dict = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "metadata_json": str(metadata_json),
        "command_txt": str(command_txt),
        "summary_md": str(summary_md),
    }

    if resource_logs:
        return_dict["resources_json"] = str(resources_json)

    return return_dict


def _format_optional_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"
