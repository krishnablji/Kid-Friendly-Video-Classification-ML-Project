"""
train.py
End-to-End Fine-Tuning Pipeline for Local DeBERTa-v3 Transformer Classifier:
- Loads audited SafeWatch-Bench data splits (safewatch_train.csv, safewatch_val.csv, safewatch_test.csv).
- Fine-tunes microsoft/deberta-v3-base locally via PyTorch with CrossEntropyLoss and AdamW.
- Evaluates on validation and test sets (Accuracy, Precision, Recall, Macro F1).
- Saves fine-tuned model weights to models/deberta_kid_safe/.
- 100% offline, local compute with zero external AI API dependencies.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import (
    DEFAULT_MODEL_NAME,
    MAX_TOKEN_LENGTH,
    LABEL2ID,
    ID2LABEL,
    KidSafeTextDataset,
    get_tokenizer,
    get_model,
    get_optimizer,
    get_loss_fn,
    save_transformer_model,
)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Computes standard classification evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm
    }


def evaluate(model, dataloader, loss_fn, device: str) -> tuple[float, dict, list[int], list[float]]:
    """Evaluates model performance over a DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_prob_unsafe = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_prob_unsafe.extend(probs[:, 1].cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_prob_unsafe


def train(
    data_dir: str = "data",
    output_dir: str = "models/deberta_kid_safe",
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    train_samples: int = 500,
    eval_samples: int = 200,
    device: str | None = None
):
    print("=" * 80)
    print(" Local DeBERTa-v3 Transformer Fine-Tuning Pipeline (Stage 3)")
    print(" 100% Offline PyTorch Training | Zero External API Dependencies")
    print("=" * 80)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Compute Device: {device.upper()}")

    # 1. Load data
    train_csv = os.path.join(data_dir, "safewatch_train.csv")
    val_csv = os.path.join(data_dir, "safewatch_val.csv")
    test_csv = os.path.join(data_dir, "safewatch_test.csv")

    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required dataset split missing at: {p}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    def sample_balanced(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        if n_samples <= 0 or n_samples >= len(df):
            return df
        safe_df = df[df["label"] == "SAFE"]
        unsafe_df = df[df["label"] == "UNSAFE"]
        half = n_samples // 2
        s_safe = safe_df.sample(n=min(len(safe_df), half), random_state=42)
        s_unsafe = unsafe_df.sample(n=min(len(unsafe_df), half), random_state=42)
        return pd.concat([s_safe, s_unsafe]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_df = sample_balanced(train_df, train_samples)
    val_df = sample_balanced(val_df, eval_samples)
    test_df = sample_balanced(test_df, eval_samples)

    print(f"\nLoaded Audited Dataset Splits:")
    print(f"  Training Split:   {len(train_df):,} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"  Validation Split: {len(val_df):,} samples ({val_df['label'].value_counts().to_dict()})")
    print(f"  Testing Split:    {len(test_df):,} samples ({test_df['label'].value_counts().to_dict()})")

    # 2. Tokenizer & Datasets
    print(f"\nLoading Tokenizer: {model_name} (Max Tokens: {max_length})...")
    tokenizer = get_tokenizer(model_name)

    train_dataset = KidSafeTextDataset(
        texts=train_df["description"].tolist(),
        labels=[LABEL2ID[lbl] for lbl in train_df["label"]],
        tokenizer=tokenizer,
        max_length=max_length
    )
    val_dataset = KidSafeTextDataset(
        texts=val_df["description"].tolist(),
        labels=[LABEL2ID[lbl] for lbl in val_df["label"]],
        tokenizer=tokenizer,
        max_length=max_length
    )
    test_dataset = KidSafeTextDataset(
        texts=test_df["description"].tolist(),
        labels=[LABEL2ID[lbl] for lbl in test_df["label"]],
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer
    print(f"Initializing Backbone Model: {model_name}...")
    model = get_model(model_name, num_labels=2).to(device)
    if device == "cpu":
        model = model.float()

    # Balanced class weights
    label_counts = train_df["label"].value_counts()
    weight_safe = len(train_df) / (2.0 * label_counts.get("SAFE", len(train_df) / 2))
    weight_unsafe = len(train_df) / (2.0 * label_counts.get("UNSAFE", len(train_df) / 2))
    class_weights = torch.tensor([weight_safe, weight_unsafe], dtype=torch.float32).to(device)

    loss_fn = get_loss_fn(class_weights=class_weights)
    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)

    # 4. Training Loop
    print(f"\nStarting Fine-Tuning: Epochs={epochs}, LR={lr}, BatchSize={batch_size}, WeightDecay={weight_decay}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        t_epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

            if step % max(1, len(train_loader) // 5) == 0 or step == len(train_loader):
                cur_loss = running_loss / step_count
                print(f"  Epoch [{epoch}/{epochs}] | Step [{step}/{len(train_loader)}] | Batch Loss: {cur_loss:.4f}")

        # Validation Step
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, loss_fn, device)
        epoch_sec = time.time() - t_epoch_start
        print(f"\n--- Epoch {epoch} Summary ({epoch_sec:.1f}s) ---")
        print(f"  Val Loss: {val_loss:.4f} | Accuracy: {val_metrics['accuracy']*100:.2f}% | Macro F1: {val_metrics['macro_f1']*100:.2f}%")
        print(f"  Val Confusion Matrix:\n{val_metrics['confusion_matrix']}")

    total_train_sec = time.time() - start_time
    print(f"\nFine-Tuning Finished in {total_train_sec:.2f} seconds!")

    # 5. Final Evaluation on Held-Out Test Set
    print("\nRunning Evaluation on Held-Out Test Set...")
    test_loss, test_metrics, test_preds, test_prob_unsafe = evaluate(model, test_loader, loss_fn, device)

    print("\n" + "=" * 50)
    print(" HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Test Loss:        {test_loss:.4f}")
    print(f"  Test Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"  Macro Precision:  {test_metrics['precision']*100:.2f}%")
    print(f"  Macro Recall:     {test_metrics['recall']*100:.2f}%")
    print(f"  Macro F1-Score:   {test_metrics['macro_f1']*100:.2f}%")
    print(f"  Confusion Matrix (Row=Actual [SAFE, UNSAFE], Col=Pred):\n{test_metrics['confusion_matrix']}")
    print("=" * 50)

    # 6. Save Model & Artifacts
    save_transformer_model(model, tokenizer, output_dir)

    # Save test predictions CSV
    test_out_df = test_df.copy()
    test_out_df["predicted_label"] = [ID2LABEL[p] for p in test_preds]
    test_out_df["prob_unsafe"] = [round(p, 4) for p in test_prob_unsafe]
    test_preds_path = os.path.join(data_dir, "test_predictions.csv")
    test_out_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions to: {test_preds_path}")

    return {
        "model_dir": output_dir,
        "test_metrics": test_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune local DeBERTa-v3 Transformer on SafeWatch-Bench")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with CSV splits")
    parser.add_argument("--output_dir", type=str, default="models/deberta_kid_safe", help="Output directory for model weights")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model backbone")
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length")
    parser.add_argument("--train_samples", type=int, default=300, help="Number of training samples (use -1 for full)")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of eval samples (use -1 for full)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples
    )


if __name__ == "__main__":
    main()
