import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import json
import os
from dataset import create_sample_datasets, compute_metrics
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# SingleTaskDataset Class - handles samples for one task at a time (Sec 3.2.5)
class SingleTaskDataset(Dataset):
    """
    Dataset class for a single task.
    Each sample contains input text and corresponding label.
    Used with task-aware batch scheduling where each batch
    is drawn from a single task dataset.
    """

    def __init__(self, data: List[Tuple], tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

#Class MultitaskModel model architecture - Input Text -> Shared Encoder -> Task-Specific Head -> Task Prediction
class MultitaskModel(nn.Module):
    """
    Multitask model with shared encoder and task-specific heads.

    Architecture:
    - Shared transformer encoder (BERT) (Sec 3.2.3)
    - Task-specific classification heads (Sec 3.2.4)
    - Binary tasks: y = sigma(Wz + b) with single output (Sec 3.2.4.1-2)
    - Multi-class tasks: y = softmax(Wz + b) with num_classes outputs (Sec 3.2.4.3)
    """

    def __init__(self, model_name: str, task_configs: Dict[str, int]):
        super().__init__()

        # Shared encoder - using BERT (Sec 3.2.3)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Task-specific heads: y = Wz + b (Sec 3.2.4)
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            if num_classes == 2:
                # Binary classification: sigma(Wz + b) -> single output (Sec 3.2.4.1-2)
                self.task_heads[task_name] = nn.Linear(hidden_size, 1)
            else:
                # Multi-class classification: softmax(Wz + b) -> num_classes outputs (Sec 3.2.4.3)
                self.task_heads[task_name] = nn.Linear(hidden_size, num_classes)

        self.task_configs = task_configs

    #one input flows through shared layers then splits to different heads.
    def forward(self, input_ids, attention_mask, task_name):
        # Get shared representations from encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation for classification (z = h_[CLS], Sec 3.2.3)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Pass through task-specific head
        logits = self.task_heads[task_name](pooled_output)

        return logits

#class multitask trainer
class MultitaskTrainer:
    """
    Trainer class for multitask learning with the following features:
    1. Task-aware batch scheduling (Sec 3.2.5)
    2. Loss balancing with task weights (Sec 3.3.3)
    3. Gradient clipping
    4. AdamW optimizer (Sec 3.3.4)
    """

    def __init__(self, model, tokenizer, task_configs, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.task_configs = task_configs
        self.device = device

        # Initialize AdamW optimizer (Sec 3.3.4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)

        # Loss functions: BCE for binary tasks, CE for multi-class (Sec 3.3.1-2)
        self.loss_fns = {}
        for task, num_classes in task_configs.items():
            if num_classes == 2:
                self.loss_fns[task] = nn.BCEWithLogitsLoss()
            else:
                self.loss_fns[task] = nn.CrossEntropyLoss()

        # Task weights for loss balancing (lambda = 1 for all, Sec 3.3.3)
        self.task_weights = {task: 1.0 for task in task_configs.keys()}

        # Training history
        self.history = {
            'train_loss': [],
            'task_losses': {task: [] for task in task_configs.keys()},
            'val_metrics': {task: [] for task in task_configs.keys()}
        }

        self.start_epoch = 0  # For resuming from checkpoint

    def save_checkpoint(self, epoch, checkpoint_dir="checkpoints", mid_epoch=False):
        """Save a checkpoint so training can be resumed."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'mid_epoch': mid_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        if not mid_epoch:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, path)
            print(f"  Checkpoint saved: {path}")
        # Always save as "latest" for easy resume
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        if mid_epoch:
            print(f"  Mid-epoch checkpoint saved")

    def load_checkpoint(self, checkpoint_dir="checkpoints"):
        """Load the latest checkpoint to resume training. Returns True if resumed."""
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(latest_path):
            return False

        print(f"\nFound checkpoint at {latest_path}, resuming training...")
        checkpoint = torch.load(latest_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        # If mid-epoch, restart that epoch (with updated model weights)
        if checkpoint.get('mid_epoch', False):
            self.start_epoch = checkpoint['epoch']
            print(f"  Resumed mid-epoch {checkpoint['epoch'] + 1}. Will restart epoch {self.start_epoch + 1}.")
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"  Resumed after epoch {checkpoint['epoch'] + 1}. Will continue from epoch {self.start_epoch + 1}.")
        return True

    #Training Loop - Task-aware Batch Scheduling (Sec 3.2.5)
    #Each batch is drawn from a single task dataset, alternating between tasks
    def train_epoch(self, task_dataloaders, epoch):
        """Train for one epoch with task-aware batch scheduling"""
        self.model.train()
        total_loss = 0
        task_losses = {task: 0 for task in self.task_configs.keys()}
        task_counts = {task: 0 for task in self.task_configs.keys()}

        # Create iterators for each task dataloader
        task_iters = {task: iter(dl) for task, dl in task_dataloaders.items()}
        task_names = list(task_dataloaders.keys())
        total_steps = sum(len(dl) for dl in task_dataloaders.values())

        step = 0
        task_idx = 0
        exhausted = set()

        # Alternate between tasks in round-robin fashion
        while step < total_steps and len(exhausted) < len(task_names):
            task_name = task_names[task_idx % len(task_names)]
            task_idx += 1

            if task_name in exhausted:
                continue

            try:
                batch = next(task_iters[task_name])
            except StopIteration:
                exhausted.add(task_name)
                continue

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass through task-specific head
            logits = self.model(input_ids, attention_mask, task_name)

            # Compute task-specific loss (Sec 3.3.1-2)
            if self.task_configs[task_name] == 2:
                # Binary: BCE loss with sigmoid (Sec 3.3.1)
                loss = self.loss_fns[task_name](logits.squeeze(-1), labels.float())
            else:
                # Multi-class: CE loss with softmax (Sec 3.3.2)
                loss = self.loss_fns[task_name](logits, labels)

            weighted_loss = loss * self.task_weights[task_name]

            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += weighted_loss.item()
            task_losses[task_name] += loss.item()
            task_counts[task_name] += 1
            step += 1

            # Print progress
            if step % 10 == 0:
                print(f'Epoch {epoch}, Step {step}/{total_steps}, Task: {task_name}, Loss: {loss.item():.4f}')

            # Mid-epoch checkpoint every 1000 steps
            if step % 1000 == 0:
                self.save_checkpoint(epoch, checkpoint_dir="checkpoints", mid_epoch=True)

        # Average losses
        avg_total_loss = total_loss / max(step, 1)
        avg_task_losses = {task: (task_losses[task] / max(task_counts[task], 1))
                          for task in self.task_configs.keys()}

        # Update history
        self.history['train_loss'].append(avg_total_loss)
        for task, loss in avg_task_losses.items():
            self.history['task_losses'][task].append(loss)

        return avg_total_loss, avg_task_losses

    #separate evaluation - a model might be good at sentiment but bad at intent classification.
    def evaluate(self, task_dataloaders):
        """Evaluate the model on validation data"""
        self.model.eval()
        task_predictions = {task: [] for task in self.task_configs.keys()}
        task_labels = {task: [] for task in self.task_configs.keys()}

        with torch.no_grad():
            for task_name, dataloader in task_dataloaders.items():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label']

                    # Forward pass
                    logits = self.model(input_ids, attention_mask, task_name)

                    if self.task_configs[task_name] == 2:
                        # Binary: sigmoid + threshold at 0.5 (Sec 3.2.6)
                        probs = torch.sigmoid(logits.squeeze(-1))
                        predictions = (probs > 0.5).long()
                    else:
                        # Multi-class: argmax (Sec 3.2.6)
                        predictions = torch.argmax(logits, dim=-1)

                    task_predictions[task_name].extend(predictions.cpu().numpy())
                    task_labels[task_name].extend(labels.numpy())

        # Compute metrics for each task
        task_metrics = {}
        for task in self.task_configs.keys():
            if len(task_predictions[task]) > 0:
                metrics = compute_metrics(task_predictions[task], task_labels[task])
                task_metrics[task] = metrics
                self.history['val_metrics'][task].append(metrics)

        return task_metrics

def main():
    """
    Main training function demonstrating the complete multitask fine-tuning pipeline
    """
    print(" Starting Multitask Fine-tuning Tutorial")
    print("=" * 50)

    # Configuration
    MODEL_NAME = "bert-base-uncased"  # Shared BERT encoder (Sec 3.2.3)
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    MAX_LENGTH = 128
    # MAX_SAMPLES = 1000  # Set to None to use full dataset

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Task configurations (task_name: num_classes)
    task_configs = {
        'sarc': 2,      # non-sarcastic, sarcastic
        'intent': 2,     # not bullying, bullying
        'emotion': 6     # sad, joy, love, angry, fear, surprise
    }

    print(f"\nTask configurations:")
    for task, num_classes in task_configs.items():
        print(f"  - {task}: {num_classes} classes")

    # Load tokenizer
    print(f"\n Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create sample datasets
    print("\n Creating sample datasets...")
    tasks_data = create_sample_datasets()

    # Limit dataset size for testing
    # if MAX_SAMPLES is not None:
    #     for task_name in tasks_data:
    #         tasks_data[task_name] = tasks_data[task_name][:MAX_SAMPLES]

    # Print dataset statistics
    print("\nDataset statistics:")
    for task_name, task_data in tasks_data.items():
        print(f"  - {task_name}: {len(task_data)} samples")

    # Split data into train/validation/test (80/10/10 split, Sec 3.2.1.2)
    train_data = {}
    val_data = {}
    test_data = {}

    for task_name, task_samples in tasks_data.items():
        random.shuffle(task_samples)
        split_train = int(0.8 * len(task_samples))
        split_val = int(0.9 * len(task_samples))
        train_data[task_name] = task_samples[:split_train]
        val_data[task_name] = task_samples[split_train:split_val]
        test_data[task_name] = task_samples[split_val:]

    # Create per-task DataLoaders (Sec 3.2.5 - Task-aware Batch Scheduling)
    print("\n Creating per-task PyTorch datasets...")
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for task_name in tasks_data:
        train_ds = SingleTaskDataset(train_data[task_name], tokenizer, MAX_LENGTH)
        val_ds = SingleTaskDataset(val_data[task_name], tokenizer, MAX_LENGTH)
        test_ds = SingleTaskDataset(test_data[task_name], tokenizer, MAX_LENGTH)
        train_loaders[task_name] = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loaders[task_name] = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loaders[task_name] = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    total_train = sum(len(train_data[t]) for t in tasks_data)
    total_val = sum(len(val_data[t]) for t in tasks_data)
    total_test = sum(len(test_data[t]) for t in tasks_data)
    print(f"Train dataset size: {total_train}")
    print(f"Validation dataset size: {total_val}")
    print(f"Test dataset size: {total_test}")

    # Initialize model
    print(f"\n Initializing multitask model...")
    model = MultitaskModel(MODEL_NAME, task_configs)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize trainer
    trainer = MultitaskTrainer(model, tokenizer, task_configs, device)

    # Try to resume from checkpoint
    CHECKPOINT_DIR = "checkpoints"
    trainer.load_checkpoint(CHECKPOINT_DIR)

    # Training loop
    print(f"\n Starting training for {NUM_EPOCHS} epochs...")
    if trainer.start_epoch > 0:
        print(f"  (Skipping epochs 1-{trainer.start_epoch}, already completed)")
    print("-" * 50)

    for epoch in range(trainer.start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        # Train with task-aware batch scheduling
        train_loss, task_losses = trainer.train_epoch(train_loaders, epoch)
        print(f"Training - Overall Loss: {train_loss:.4f}")
        for task, loss in task_losses.items():
            print(f"  {task}: {loss:.4f}")

        # Evaluate
        val_metrics = trainer.evaluate(val_loaders)
        print(f"\nValidation Results:")
        for task, metrics in val_metrics.items():
            print(
                f"  {task}: Accuracy={metrics['accuracy']:.3f}, "
                f"Precision={metrics['precision']:.3f}, "
                f"Recall={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}"
            )

        # Save checkpoint after each epoch
        trainer.save_checkpoint(epoch, CHECKPOINT_DIR)

    print("\nTraining completed!")

    # === Save Results ===
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save trained model
    model_path = os.path.join(results_dir, "mtl_bert_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # 2. Save training history (losses + validation metrics) as JSON
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # 3. Final test evaluation
    print(f"\nFinal Test Evaluation:")
    print("-" * 50)
    test_metrics = trainer.evaluate(test_loaders)
    test_results = {}
    for task, metrics in test_metrics.items():
        test_results[task] = metrics
        print(
            f"  {task}: Accuracy={metrics['accuracy']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}"
        )

    # Save test metrics as JSON
    test_path = os.path.join(results_dir, "test_metrics.json")
    with open(test_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test metrics saved to {test_path}")

    # 4. Plot and save training loss curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot total training loss
    axes[0].plot(range(1, NUM_EPOCHS + 1), trainer.history['train_loss'], marker='o', label='Total Loss')
    for task in task_configs:
        axes[0].plot(range(1, NUM_EPOCHS + 1), trainer.history['task_losses'][task], marker='s', label=f'{task} Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss per Epoch')
    axes[0].legend()
    axes[0].grid(True)

    # Plot validation F1 scores per task
    for task in task_configs:
        f1_scores = [m['f1'] for m in trainer.history['val_metrics'][task]]
        axes[1].plot(range(1, len(f1_scores) + 1), f1_scores, marker='o', label=f'{task} F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score per Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "training_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training plots saved to {plot_path}")

    print(f"\nAll results saved to '{results_dir}/' directory.")

    #inference
    print("Demonstration of inference on new samples:")
    print("-" * 50)

    test_samples = [
        ("This movie is absolutely fantastic!", "sarc"),
        ("Even as a troll you are a pathetic failure.", "intent"),
        ("i drove into the premises of the school the feeling was strange", "emotion")
    ]

    model.eval()
    with torch.no_grad():
        for text, expected_task in test_samples:
            # Tokenize
            encoding = tokenizer(text, truncation=True, padding='max_length',
                               max_length=MAX_LENGTH, return_tensors='pt')

            # Predict
            logits = model(encoding['input_ids'].to(device),
                          encoding['attention_mask'].to(device),
                          expected_task)

            if task_configs[expected_task] == 2:
                # Binary task: sigmoid activation (Sec 3.2.4.1-2)
                prob = torch.sigmoid(logits.squeeze(-1)).item()
                prediction = 1 if prob > 0.5 else 0
                confidence = prob if prediction == 1 else 1 - prob
            else:
                # Multi-class: softmax activation (Sec 3.2.4.3)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()

            print(f"Text: '{text}'")
            print(f"Task: {expected_task}")
            print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
            print()

if __name__ == "__main__":
    print("=" * 50)
    main()
