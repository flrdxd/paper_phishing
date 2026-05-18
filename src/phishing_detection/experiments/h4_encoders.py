"""H4 experiment: compact transformer encoders for MeAJOR."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from phishing_detection.experiments.datasets import build_dataset_metadata, load_meajor_dataframe, split_random
from phishing_detection.experiments.metrics import calculate_research_metrics
from phishing_detection.experiments.results import save_experiment_run


logger = logging.getLogger(__name__)

ENCODER_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "minilm": "microsoft/MiniLM-L12-H384-uncased",
    "bert": "bert-base-uncased",
}


class TextDataset(Dataset):
    """Small text classification dataset for transformer experiments."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


def run_h4_compact_encoders(
    sample_size: int | None = 1000,
    models: str = "distilbert,minilm",
    epochs: int = 1,
    max_length: int = 256,
    batch_size: int = 8,
    force_download: bool = False,
    random_state: int = 42,
) -> dict:
    """Run compact encoder fine-tuning on MeAJOR."""
    start_time = time.time()
    selected_models = parse_encoder_models(models)
    df = load_meajor_dataframe(sample_size, random_state, force_download)
    train_df, test_df = split_random(df, random_state=random_state)
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=random_state,
        stratify=train_df["label"],
    )

    metadata = build_dataset_metadata(df)
    metadata.update({
        "split_policy": "random",
        "random_state": random_state,
        "epochs": epochs,
        "max_length": max_length,
        "batch_size": batch_size,
        "hypothesis": "H4 compact encoders can approach larger text encoders",
    })

    results = []
    for model_key in selected_models:
        result = train_encoder(
            model_key=model_key,
            hf_model_name=ENCODER_MODELS[model_key],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            epochs=epochs,
            max_length=max_length,
            batch_size=batch_size,
            random_state=random_state,
        )
        results.append(result)

    metadata["total_time_seconds"] = time.time() - start_time
    outputs = save_experiment_run(
        experiment_name="h4_compact_encoders",
        results=results,
        metadata=metadata,
        command=build_h4_command(sample_size, models, epochs, max_length, batch_size, force_download, random_state),
        summary_title="H4 Compact Encoders",
    )
    return {"results": results, "metadata": metadata, "output_paths": outputs}


def parse_encoder_models(models: str) -> list[str]:
    selected = [model.strip().lower() for model in models.split(",") if model.strip()]
    unknown = sorted(set(selected) - set(ENCODER_MODELS))
    if unknown:
        raise ValueError(f"Unknown encoder(s): {unknown}. Available: {sorted(ENCODER_MODELS)}")
    return selected or ["distilbert", "minilm"]


def train_encoder(model_key, hf_model_name, train_df, val_df, test_df, epochs, max_length, batch_size, random_state):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training encoder %s on %s", model_key, device)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=2)
    model.to(device)

    train_loader = DataLoader(TextDataset(train_df["text"], train_df["label"], tokenizer, max_length), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TextDataset(test_df["text"], test_df["label"], tokenizer, max_length), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_start = time.time()
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                logger.info("%s epoch %s/%s batch %s/%s loss=%.4f", model_key, epoch + 1, epochs, batch_index, len(train_loader), np.mean(losses))
    training_time = time.time() - train_start

    infer_start = time.time()
    y_pred, y_scores, y_true = evaluate_encoder(model, test_loader, device)
    inference_time = time.time() - infer_start
    metrics = calculate_research_metrics(y_true, y_pred, y_scores)
    metrics.update({
        "model": model_key,
        "hf_model_name": hf_model_name,
        "training_time": training_time,
        "inference_time": inference_time,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "device": str(device),
    })
    return metrics


def evaluate_encoder(model, data_loader, device):
    model.eval()
    predictions = []
    scores = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
            scores.extend(probabilities[:, 1].cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    return np.array(predictions), np.array(scores), np.array(labels)


def build_h4_command(sample_size, models, epochs, max_length, batch_size, force_download, random_state) -> str:
    parts = [
        "python3 run.py experiment h4-compact-encoders",
        f"--models {models}",
        f"--epochs {epochs}",
        f"--max-length {max_length}",
        f"--batch-size {batch_size}",
        f"--random-state {random_state}",
    ]
    if sample_size:
        parts.append(f"--sample-size {sample_size}")
    if force_download:
        parts.append("--force-download")
    return " ".join(parts)
