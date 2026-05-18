"""Feature builders shared by research experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


URL_META_NUMERIC_COLUMNS = [
    "url_count",
    "url_length_max",
    "url_length_avg",
    "url_subdom_max",
    "url_subdom_avg",
    "attachment_count",
]

URL_META_CATEGORICAL_COLUMNS = [
    "has_attachments",
    "sender_domain",
    "receiver_domain",
    "content_types",
    "language",
]


def available_url_meta_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical URL/metadata columns present in a frame."""
    numeric = [column for column in URL_META_NUMERIC_COLUMNS if column in df.columns]
    categorical = [column for column in URL_META_CATEGORICAL_COLUMNS if column in df.columns]
    return numeric, categorical


def build_text_preprocessor(max_features: int = 5000) -> TfidfVectorizer:
    """Create the standard text TF-IDF preprocessor."""
    return TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words="english",
    )


def build_url_meta_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for URL and metadata features."""
    numeric_columns, categorical_columns = available_url_meta_columns(df)
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("stringify", FunctionTransformer(lambda frame: frame.astype(str))),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No URL/metadata columns available for this dataset")
    return ColumnTransformer(transformers=transformers)


def build_text_url_meta_preprocessor(df: pd.DataFrame, max_features: int = 5000) -> ColumnTransformer:
    """Create preprocessing for combined text and URL/metadata features."""
    numeric_columns, categorical_columns = available_url_meta_columns(df)
    transformers = [
        ("text", build_text_preprocessor(max_features=max_features), "text"),
    ]
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("stringify", FunctionTransformer(lambda frame: frame.astype(str))),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers)


def get_model_scores(pipeline, frame_or_texts):
    """Return positive-class scores from a fitted sklearn pipeline."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(frame_or_texts)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(frame_or_texts)
    return None
