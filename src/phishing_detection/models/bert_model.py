"""
BERT Model for Phishing Email Detection

This module implements BERT (Bidirectional Encoder Representations from Transformers)
fine-tuning for binary classification of phishing emails.

Key features:
- Uses bert-base-uncased from Hugging Face
- Fine-tuned for binary classification
- Supports mixed precision training
- Early stopping to prevent overfitting
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import gc

# Import path configuration
from phishing_detection.path_config import PATHS

# Set CUDA memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class EmailDataset(Dataset):
    """Dataset class for email texts."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize email dataset.

        Args:
            texts: List of email texts
            labels: List of labels (0=legitimate, 1=phishing)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
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
    """BERT classifier for phishing email detection."""

    def __init__(self, model_name='bert-base-uncased', max_length=512, random_state=42):
        """
        Initialize BERT detector.

        Args:
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length
            random_state: Random state for reproducibility
        """
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.tokenizer = None
        self.model = None
        self.training_time = None
        self.inference_time = None
        self.mixed_precision = torch.cuda.is_available()

        # Training hyperparameters (aggressive for 100+ GB VRAM - matches paper)
        self.learning_rate = 2e-5
        self.batch_size = 16  # Paper settings for max speed/accuracy
        self.eval_batch_size = 64  # Paper settings
        self.epochs = 10
        self.early_stopping_patience = 3

        torch.manual_seed(random_state)
        np.random.seed(random_state)

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
        """
        Prepare data loaders for training.

        Args:
            X_train: Training texts
            X_test: Test texts
            y_train: Training labels
            y_test: Test labels

        Returns:
            tuple: Train, validation, and test dataloaders
        """
        # Split training data for validation (80/20)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train
        )

        # Create datasets
        train_dataset = EmailDataset(X_train_split.values, y_train_split.values, self.tokenizer, self.max_length)
        val_dataset = EmailDataset(X_val.values, y_val.values, self.tokenizer, self.max_length)
        test_dataset = EmailDataset(X_test.values, y_test.values, self.tokenizer, self.max_length)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.eval_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch_size, shuffle=False)

        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler, scaler):
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        return total_loss / len(train_loader)

    def evaluate(self, data_loader):
        """
        Evaluate the model.

        Args:
            data_loader: Data loader

        Returns:
            tuple: Average loss, predictions, true labels
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        return avg_loss, all_predictions, all_labels

    def train(self, X_train, X_test, y_train, y_test):
        """
        Train the BERT model.

        Args:
            X_train: Training texts
            X_test: Test texts
            y_train: Training labels
            y_test: Test labels

        Returns:
            self: Trained model
        """
        logger.info("Starting BERT training...")
        logger.info(f"Mixed precision enabled: {self.mixed_precision}")

        # Load model
        self.load_model()

        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(X_train, X_test, y_train, y_test)

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, scaler)
            training_losses.append(train_loss)

            # Validate
            val_loss, val_predictions, val_labels = self.evaluate(val_loader)
            val_losses.append(val_loss)

            # Calculate validation accuracy
            val_accuracy = accuracy_score(val_labels, val_predictions)

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(PATHS['MODELS_DIR'], 'bert_temp_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(PATHS['MODELS_DIR'], 'bert_temp_best.pth')))
        os.remove(os.path.join(PATHS['MODELS_DIR'], 'bert_temp_best.pth'))

        self.training_time = time.time() - start_time
        logger.info(f"\nTraining complete in {self.training_time:.2f} seconds ({self.training_time / 60:.2f} minutes)")

        return self

    def predict(self, X_test, y_test):
        """
        Make predictions on test data.

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating on test set...")

        test_dataset = EmailDataset(X_test.values, y_test.values, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch_size, shuffle=False)

        start_time = time.time()
        test_loss, predictions, labels = self.evaluate(test_loader)
        self.inference_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Classification report
        report = classification_report(labels, predictions, target_names=['Legitimate', 'Phishing'])

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_loss': test_loss,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'device': str(device),
            'mixed_precision': self.mixed_precision,
            'epochs_configured': self.epochs,
            'confusion_matrix': cm,
            'classification_report': report
        }

        logger.info(f"\nTest Results:")
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")

        return metrics

    def save_model(self, filepath):
        """
        Save trained model to file.

        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'random_state': self.random_state,
            'training_time': self.training_time
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_saved_model(self, filepath):
        """
        Load trained model from file.

        Args:
            filepath: Path to load the model from

        Returns:
            self: Loaded model
        """
        model_data = torch.load(filepath)

        self.tokenizer = model_data['tokenizer']
        self.model_name = model_data['model_name']
        self.max_length = model_data['max_length']
        self.random_state = model_data['random_state']
        self.training_time = model_data['training_time']

        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(device)

        logger.info(f"Model loaded from {filepath}")

        return self


def train_bert(X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Convenience function to train and evaluate BERT model.

    Args:
        X_train: Training texts
        X_test: Test texts
        y_train: Training labels
        y_test: Test labels

    Returns:
        tuple: Trained model and evaluation metrics
    """
    if X_train is None or X_test is None or y_train is None or y_test is None:
        # If data not provided, load and preprocess
        from phishing_detection.data_preprocessing import DataPreprocessor

        logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor(max_features=5000, random_state=42)
        df = preprocessor.download_dataset()
        df = preprocessor.preprocess_dataframe(df)
        X_train, X_test, y_train, y_test = preprocessor.split_for_transformer(df, test_size=0.2)

    # Initialize and train model
    bert_detector = BERTPhishingDetector(model_name='bert-base-uncased', max_length=512, random_state=42)
    bert_detector.train(X_train, X_test, y_train, y_test)

    # Evaluate model
    metrics = bert_detector.predict(X_test, y_test)

    # Save model
    model_path = os.path.join(PATHS['MODELS_DIR'], 'bert_model.pth')
    bert_detector.save_model(model_path)

    logger.info("\n" + "="*50)
    logger.info("BERT MODEL RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"\nTraining Time:    {metrics['training_time']:.2f} seconds ({metrics['training_time'] / 60:.2f} minutes)")
    logger.info(f"Inference Time:   {metrics['inference_time']:.2f} seconds")
    logger.info("="*50)

    return bert_detector, metrics


if __name__ == "__main__":
    model, metrics = train_bert()
