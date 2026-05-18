"""Dataset loading helpers for research experiments."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from phishing_detection.path_config import PATHS


logger = logging.getLogger(__name__)

MEAJOR_RECORD_URL = "https://zenodo.org/records/18471483"
MEAJOR_CSV_URL = (
    "https://zenodo.org/api/records/18471483/files/"
    "meajor_cleaned_preprocessed.csv/content"
)
MEAJOR_CSV_MD5 = "aa8f59e96787cbd696c0b650e5400dc9"
MEAJOR_MIN_BYTES = 100_000_000


def get_meajor_data_dir() -> Path:
    """Return the local MeAJOR artifact directory."""
    data_dir = Path(PATHS["DATA_DIR"]) / "meajor"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_meajor_csv(force_download: bool = False) -> Path:
    """Download the MeAJOR CSV from Zenodo if it is not cached locally."""
    output_path = get_meajor_data_dir() / "meajor_cleaned_preprocessed.csv"
    if output_path.exists() and not force_download:
        logger.info("MeAJOR CSV already exists. Using cached file: %s", output_path)
        return output_path

    logger.info("Downloading MeAJOR v2.0 CSV from %s", MEAJOR_RECORD_URL)
    temp_path = output_path.with_suffix(".csv.part")
    response = requests.get(MEAJOR_CSV_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0) or 0)
    downloaded = 0
    hasher = hashlib.md5()

    with temp_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)
            hasher.update(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                logger.info("MeAJOR download progress: %.1f%%", pct)

    if downloaded < MEAJOR_MIN_BYTES:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded MeAJOR file is unexpectedly small: {downloaded} bytes"
        )

    file_md5 = hasher.hexdigest()
    if file_md5 != MEAJOR_CSV_MD5:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            "MeAJOR checksum mismatch: "
            f"expected {MEAJOR_CSV_MD5}, got {file_md5}"
        )

    temp_path.replace(output_path)
    logger.info("MeAJOR CSV saved to %s", output_path)
    return output_path


def normalize_meajor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MeAJOR columns into text, label, source, and metadata fields."""
    required_columns = {"subject", "body", "label", "source"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"MeAJOR dataframe missing required columns: {sorted(missing)}")

    normalized = df.copy()
    normalized["subject"] = normalized["subject"].fillna("").astype(str)
    normalized["body"] = normalized["body"].fillna("").astype(str)
    normalized["text"] = (
        normalized["subject"].str.strip() + "\n\n" + normalized["body"].str.strip()
    ).str.strip()

    normalized["label"] = pd.to_numeric(normalized["label"], errors="coerce")
    normalized = normalized.dropna(subset=["label"])
    normalized["label"] = normalized["label"].astype(int)
    normalized = normalized[normalized["label"].isin([0, 1])]

    normalized["source"] = normalized["source"].fillna("unknown").astype(str)
    normalized = normalized[normalized["text"].str.len() > 0].reset_index(drop=True)
    return normalized


def load_meajor_dataframe(
    sample_size: int | None = None,
    random_state: int = 42,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load, normalize, and optionally sample the MeAJOR dataset."""
    csv_path = download_meajor_csv(force_download=force_download)
    logger.info("Loading MeAJOR CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    df = normalize_meajor_dataframe(df)

    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        if sample_size < df["label"].nunique():
            raise ValueError(
                "sample_size must be large enough to include every label class"
            )
        _, sampled_df = train_test_split(
            df,
            test_size=sample_size,
            random_state=random_state,
            stratify=df["label"],
        )
        df = sampled_df.reset_index(drop=True)

    logger.info(
        "MeAJOR loaded: samples=%s, labels=%s, sources=%s",
        len(df),
        df["label"].value_counts().to_dict(),
        df["source"].nunique(),
    )
    return df


def build_dataset_metadata(df: pd.DataFrame) -> dict:
    """Build a compact dataset metadata report."""
    text_duplicates = int(df["text"].duplicated().sum()) if "text" in df else 0
    return {
        "dataset": "MeAJOR Corpus",
        "version": "2.0",
        "record_url": MEAJOR_RECORD_URL,
        "samples": int(len(df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
        "sources": {
            str(source): int(count)
            for source, count in df["source"].value_counts().sort_index().items()
        },
        "source_count": int(df["source"].nunique()),
        "duplicate_texts": text_duplicates,
        "empty_texts": int((df["text"].str.len() == 0).sum()) if "text" in df else 0,
        "columns": list(df.columns),
    }


def split_random(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified random train/test split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_by_source(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by full source groups, avoiding source leakage."""
    source_labels = (
        df.groupby("source")["label"]
        .agg(lambda values: int(values.mean() >= 0.5))
        .reset_index()
    )
    if len(source_labels) < 2:
        raise ValueError("Source split requires at least two distinct sources")

    stratify = None
    if source_labels["label"].value_counts().min() >= 2:
        stratify = source_labels["label"]

    train_sources, test_sources = train_test_split(
        source_labels["source"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_source_set = set(train_sources)
    test_source_set = set(test_sources)

    overlap = train_source_set & test_source_set
    if overlap:
        raise ValueError(f"Source leakage detected: {sorted(overlap)}")

    train_df = df[df["source"].isin(train_source_set)].reset_index(drop=True)
    test_df = df[df["source"].isin(test_source_set)].reset_index(drop=True)
    _validate_split_classes(train_df, test_df)
    return train_df, test_df


def iter_leave_one_source_out(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return leave-one-source-out splits for sources whose test fold has both classes."""
    splits = []
    for source in sorted(df["source"].dropna().unique()):
        train_df = df[df["source"] != source].reset_index(drop=True)
        test_df = df[df["source"] == source].reset_index(drop=True)
        try:
            _validate_split_classes(train_df, test_df)
        except ValueError:
            logger.warning("Skipping source %s because it does not contain both classes", source)
            continue
        splits.append((str(source), train_df, test_df))
    if not splits:
        raise ValueError("No valid leave-one-source-out split contains both classes")
    return splits


def _validate_split_classes(*frames: Iterable[pd.DataFrame]) -> None:
    for index, frame in enumerate(frames, start=1):
        labels = set(frame["label"].unique())
        if labels != {0, 1}:
            raise ValueError(
                f"Split {index} must contain both classes; found labels {sorted(labels)}"
            )
