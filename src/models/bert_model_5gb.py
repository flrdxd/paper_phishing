"""
Alternative BERT Implementation for 5GB VRAM

This version uses gradient accumulation to maintain effective batch size
while using smaller actual batches to fit in 5GB VRAM.
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class EmailDataset(Dataset):
    """Dataset class for email texts."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Use tokenizer __call__ method (works with all transformers versions)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTPhishingDetector:
    """BERT classifier optimized for 5GB VRAM using gradient accumulation."""

    def __init__(self, model_name='bert-base-uncased', max_length=512, random_state=42):
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.tokenizer = None
        self.model = None
        self.training_time = None
        self.inference_time = None

        # Optimized for 5GB VRAM
        self.learning_rate = 2e-5
        self.actual_batch_size = 4  # Small batch for memory
        self.accumulation_steps = 4  # Accumulate 4 batches = effective 16
        self.eval_batch_size = 64
        self.epochs = 10
        self.early_stopping_patience = 3

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        logger.info(f"Gradient accumulation: {self.accumulation_steps} steps × batch {self.actual_batch_size} = effective {self.actual_batch_size * self.accumulation_steps}")

    def load_model(self):
        """Load pre-trained BERT model and tokenizer."""
        logger.info(f"Loading BERT model: {self.model_name}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(device)

        logger.info("Model and tokenizer loaded successfully")

    def prepare_data(self, X_train, X_test, y_train, y_test):
        """Prepare data loaders for training."""
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train
        )

        train_dataset = EmailDataset(X_train_split.values, y_train_split.values, self.tokenizer, self.max_length)
        val_dataset = EmailDataset(X_val.values, y_val.values, self.tokenizer, self.max_length)
        test_dataset = EmailDataset(X_test.values, y_test.values, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.actual_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.eval_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch_size, shuffle=False)

        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        logger.info(f"Gradient accumulation: {self.accumulation_steps} steps (effective batch: {self.actual_batch_size * self.accumulation_steps})")

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0

        optimizer.zero_grad()  # Reset gradients once at start of epoch
        accumulated_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / self.accumulation_steps  # Scale loss by accumulation steps
            loss.backward()  # Accumulate gradients

            accumulated_steps += 1

            # Only update weights every accumulation_steps
            if accumulated_steps % self.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients
                scheduler.step()

            total_loss += loss.item() * self.accumulation_steps

        # Handle remaining accumulated gradients at end of epoch
        if accumulated_steps % self.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        return total_loss / len(train_loader)

    def train(self, X_train, X_test, y_train, y_test):
        """Train BERT model with gradient accumulation."""
        logger.info("Starting BERT training with gradient accumulation...")
        self.load_model()
        train_loader, val_loader, test_loader = self.prepare_data(X_train, X_test, y_train, y_test)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        patience_counter = 0

        start_time = time.time()

        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    val_loss += outputs.loss.item()

            val_loss /= len(val_loader)

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        self.training_time = time.time() - start_time
        logger.info(f"Training complete in {self.training_time:.2f} seconds ({self.training_time / 60:.2f} minutes)")

        return self

    def predict(self, X_test, y_test):
        """Evaluate model on test data."""
        logger.info("Evaluating model...")
        self.load_model()

        # Prepare test dataset
        test_dataset = EmailDataset(X_test.values, y_test.values, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch_size, shuffle=False)

        self.model.eval()
        all_predictions = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self.inference_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Inference time: {self.inference_time:.2f} seconds")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_loss': val_loss if 'val_loss' in locals() else 0,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }

        return metrics


def train_bert_5gb(X_train=None, X_test=None, y_train=None, y_test=None):
    """Convenience function to train BERT optimized for 5GB VRAM."""
    if X_train is None or X_test is None or y_train is None or y_test is None:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from data_preprocessing import DataPreprocessor

        logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor(max_features=5000, random_state=42)
        df = preprocessor.download_dataset()
        df = preprocessor.preprocess_dataframe(df)
        X_train, X_test, y_train, y_test = preprocessor.split_for_transformer(df, test_size=0.2)

    bert_detector = BERTPhishingDetector(max_length=512, random_state=42)
    bert_detector.train(X_train, X_test, y_train, y_test)

    metrics = bert_detector.predict(X_test, y_test)

    logger.info("\n" + "="*50)
    logger.info("BERT RESULTS (5GB VRAM OPTIMIZED)")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"\nTraining Time:    {metrics['training_time']:.2f} seconds ({metrics['training_time'] / 60:.2f} minutes)")
    logger.info(f"Inference Time:   {metrics['inference_time']:.2f} seconds")
    logger.info("="*50)

    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(models_dir, 'bert_model_5gb.pth')
    torch.save({
        'model_state_dict': bert_detector.model.state_dict(),
        'tokenizer': bert_detector.tokenizer,
        'model_name': bert_detector.model_name,
        'max_length': bert_detector.max_length,
        'random_state': bert_detector.random_state
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    return bert_detector, metrics


if __name__ == "__main__":
    model, metrics = train_bert_5gb()
