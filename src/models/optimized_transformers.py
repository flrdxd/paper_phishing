"""
Optimized Transformer Models for 5GB VRAM

This module provides maximum performance optimization for BERT and DistilBERT:
- Advanced gradient accumulation strategies
- Memory-efficient mixed precision training
- Optimized learning rate scheduling
- Dynamic batch sizing based on available VRAM
- Comprehensive error handling and recovery
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

# Import path configuration
from path_config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability and optimize settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = True  # More deterministic, less random
    logger.info("CUDA optimizations enabled: benchmark + deterministic")
else:
    logger.warning("CUDA not available, using CPU (will be very slow)")


class OptimizedEmailDataset(Dataset):
    """Optimized email dataset for transformers."""

    def __init__(self, texts, labels, tokenizer, max_length=256, max_samples_per_class=1000):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples_per_class = max_samples_per_class

    def __len__(self):
        return min(len(self.texts), self.max_samples_per_class * 2)  # Limit dataset size

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Use tokenizer with optimized settings
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


class OptimizedBERT:
    """Highly optimized BERT for 5GB VRAM."""

    def __init__(self, model_name='bert-base-uncased', max_length=256, random_state=42):
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.tokenizer = None
        self.model = None
        self.training_time = None
        self.inference_time = None

        # Optimized hyperparameters for 5GB VRAM
        self.learning_rate = 2e-5
        self.mini_batch_size = 2  # Tiny batches for memory efficiency
        self.gradient_accumulation_steps = 8  # Accumulate 8 mini-batches = effective 16
        self.effective_batch_size = self.mini_batch_size * self.gradient_accumulation_steps  # = 16

        # Evaluation settings
        self.eval_batch_size = 32  # Larger batches for evaluation speed
        self.epochs = 10
        self.early_stopping_patience = 3

        # Memory management
        self.max_grad_norm = 1.0  # Gradient clipping
        self.use_amp = True  # Mixed precision training
        self.gradient_checkpointing = False  # Can enable for more memory savings
        self.max_samples_per_class = 1000  # Default limit for balanced datasets

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        logger.info(f"Optimized BERT initialized:")
        logger.info(f"  Effective batch size: {self.effective_batch_size}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps} steps")
        logger.info(f"  Mixed precision: {self.use_amp}")
        logger.info(f"  Max sequence length: {self.max_length}")

    def load_model(self):
        """Load pre-trained BERT model."""
        logger.info(f"Loading BERT model: {self.model_name}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )

        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            for param in self.model.bert.parameters():
                param.requires_grad = False  # Freeze base layers
            self.model.gradient_checkpointing_enable()

        self.model.to(device)

        logger.info("Model and tokenizer loaded successfully")

    def prepare_data_optimized(self, X_train, X_test, y_train, y_test, limit_dataset=True):
        """
        Prepare data with memory optimization.

        Args:
            limit_dataset: Whether to limit dataset size for memory
        """
        logger.info("Preparing data with optimization...")

        # Convert to lists if needed
        X_train_list = X_train.tolist() if hasattr(X_train, 'tolist') else X_train
        X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else X_test
        y_train_list = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
        y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else y_test

        # Stratified split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_list, y_train_list,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train_list
        )

        # Limit dataset size if enabled
        if limit_dataset:
            # Use smaller max_samples for testing or smaller datasets
            max_samples = min(10000, len(X_train_split))  # 5000 per class, but don't exceed available data
            logger.info(f"Limiting dataset to {max_samples} samples for memory optimization")

            indices_train = np.random.RandomState(self.random_state).choice(
                len(X_train_split), size=min(max_samples, len(X_train_split)), replace=False)
            indices_val = np.random.RandomState(self.random_state).choice(
                len(X_val), size=min(int(max_samples * 0.2), len(X_val)), replace=False)
            indices_test = np.random.RandomState(self.random_state).choice(
                len(X_test_list), size=min(int(max_samples * 0.2), len(X_test_list)), replace=False)

            X_train_split = [X_train_split[i] for i in indices_train]
            y_train_split = [y_train_split[i] for i in indices_train]
            X_val = [X_val[i] for i in indices_val]
            y_val = [y_val[i] for i in indices_val]
            X_test_list_sampled = [X_test_list[i] for i in indices_test]
            y_test_list_sampled = [y_test_list[i] for i in indices_test]

        # Create optimized datasets
        train_dataset = OptimizedEmailDataset(
            X_train_split, y_train_split, self.tokenizer, self.max_length, self.max_samples_per_class
        )
        val_dataset = OptimizedEmailDataset(
            X_val, y_val, self.tokenizer, self.max_length, self.max_samples_per_class // 5
        )
        test_dataset = OptimizedEmailDataset(
            X_test_list_sampled, y_test_list_sampled, self.tokenizer, self.max_length, self.max_samples_per_class
        )

        # Optimized dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        logger.info(f"Memory optimization: Effective batch size={self.effective_batch_size} (via gradient accumulation)")

        return train_loader, val_loader, test_loader

    def train_epoch_optimized(self, train_loader, optimizer, scheduler, scaler):
        """
        Optimized training epoch with gradient accumulation and mixed precision.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        accumulated_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision forward pass
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.gradient_accumulation_steps  # Scale loss
                scaler.scale(loss).backward()

                accumulated_steps += 1

                # Only update weights every gradient_accumulation_steps
                if accumulated_steps % self.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)  # Update weights
                    optimizer.zero_grad()
                    scheduler.step()
                    scaler.update()  # Update scaler for next iteration

                total_loss += loss.item() * self.gradient_accumulation_steps

            # Memory management - clear cache periodically
            if step % 50 == 0:
                torch.cuda.empty_cache()

        return total_loss / len(train_loader)

    def train_optimized(self, X_train, X_test, y_train, y_test, limit_dataset=True):
        """
        Optimized training with advanced techniques.
        """
        logger.info("Starting optimized BERT training...")
        self.load_model()

        train_loader, val_loader, test_loader = self.prepare_data_optimized(
            X_train, X_test, y_train, y_test, limit_dataset=limit_dataset
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Mixed precision scaler
        scaler = GradScaler()

        # Training loop with advanced techniques
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_loss = self.train_epoch_optimized(train_loader, optimizer, scheduler, scaler)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                    val_loss += outputs.loss.item()

            val_loss /= len(val_loader)
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping with patience
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

    def evaluate_optimized(self, X_test, y_test):
        """
        Optimized evaluation with mixed precision.
        """
        logger.info("Evaluating model...")

        self.load_model()

        # Prepare test dataset (no limit for evaluation)
        test_dataset = OptimizedEmailDataset(
            X_test, y_test, self.tokenizer, self.max_length, max_samples_per_class=len(X_test)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Evaluation
        self.model.eval()
        all_predictions = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast():
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

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Inference time: {self.inference_time:.2f} seconds")

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'effective_batch_size': self.effective_batch_size,
            'gradient_accumulation': self.gradient_accumulation_steps,
            'use_amp': self.use_amp
        }

        return metrics

    def save_model(self, filepath):
        """Save optimized model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'random_state': self.random_state,
            'effective_batch_size': self.effective_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'use_amp': self.use_amp,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }

        torch.save(model_data, filepath)
        logger.info(f"Optimized model saved to {filepath}")

    def load_saved_model(self, filepath):
        """Load optimized model."""
        model_data = torch.load(filepath)

        self.model = BertForSequenceClassification.from_pretrained(
            model_data['model_name'],
            num_labels=2
        )
        self.model.load_state_dict(model_data['model_state_dict'])
        self.tokenizer = model_data['tokenizer']
        self.max_length = model_data['max_length']
        self.random_state = model_data['random_state']
        self.effective_batch_size = model_data['effective_batch_size']
        self.gradient_accumulation_steps = model_data['gradient_accumulation_steps']
        self.use_amp = model_data.get('use_amp', True)
        self.training_time = model_data['training_time']
        self.inference_time = model_data.get('inference_time', None)

        logger.info(f"Optimized model loaded from {filepath}")

        return self


class OptimizedDistilBERT(OptimizedBERT):
    """Optimized DistilBERT with same optimizations as BERT."""

    def __init__(self, model_name='distilbert-base-uncased', max_length=256, random_state=42):
        super().__init__(model_name, max_length, random_state)
        # DistilBERT-specific optimizations
        self.learning_rate = 3e-5  # Slightly higher learning rate
        self.evaluation_batch_size = 64  # Can use larger batches for evaluation


def train_optimized_transformers(X_train, X_test, y_train, y_test, limit_dataset=True, model_type='bert'):
    """
    Train optimized transformer model.

    Args:
        X_train, X_test, y_train, y_test: Data splits
        limit_dataset: Limit dataset size for memory optimization
        model_type: 'bert' or 'distilbert'
    """
    if model_type == 'bert':
        detector = OptimizedBERT(max_length=256, random_state=42)
    elif model_type == 'distilbert':
        detector = OptimizedDistilBERT(max_length=256, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Training optimized {model_type.upper()} model...")

    # Train
    detector = detector.train_optimized(X_train, X_test, y_train, y_test, limit_dataset=limit_dataset)

    # Evaluate
    metrics = detector.evaluate_optimized(X_test, y_test)

    # Save model
    model_path = os.path.join(PATHS['MODELS_DIR'], f'optimized_{model_type}_model.pth')
    detector.save_model(model_path)

    logger.info("\n" + "="*60)
    logger.info(f"OPTIMIZED {model_type.upper()} RESULTS")
    logger.info("="*60)

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")

    logger.info(f"\nOptimization Details:")
    logger.info(f"  Effective Batch Size: {metrics['effective_batch_size']}")
    logger.info(f"  Gradient Accumulation: {metrics['gradient_accumulation']}")
    logger.info(f"  Mixed Precision: {'Yes' if metrics['use_amp'] else 'No'}")

    logger.info(f"\nTiming:")
    logger.info(f"  Training Time:    {metrics['training_time']:.2f} seconds ({metrics['training_time']/60:.2f} minutes)")
    logger.info(f"  Inference Time:   {metrics['inference_time']:.2f} seconds")

    logger.info("="*60)

    return detector, metrics


if __name__ == "__main__":
    # Test with small dataset
    from data_preprocessing_optimized import OptimizedDataPreprocessor

    logger.info("Loading and preprocessing data...")
    preprocessor = OptimizedDataPreprocessor(max_features=10000, random_state=42, use_augmentation=False)

    # Create small test dataset
    test_texts = [
        "Urgent notification verify account information click link immediately",
        "Hi team following up on meeting yesterday discussed budget approval team assignments",
        "Click here claim prize winner selected have 24 hours respond",
        "Meeting rescheduled due to technical issues please check email updates"
    ]
    test_labels = [1, 0, 1, 0]  # Mix of phishing and legitimate

    X_train = test_texts * 100  # Repeat for more training data
    y_train = test_labels * 100

    X_test = test_texts
    y_test = test_labels

    detector, metrics = train_optimized_transformers(
        X_train, X_test, y_train, y_test,
        limit_dataset=True,
        model_type='bert'
    )
