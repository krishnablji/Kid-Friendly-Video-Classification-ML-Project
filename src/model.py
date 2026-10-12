"""
model.py
Modern Deep Learning / Transformer Sequence Classification Module:
- Wraps microsoft/deberta-v3-base via Hugging Face AutoModelForSequenceClassification.
- Standard AutoTokenizer with max_length 512.
- CrossEntropyLoss with optional class weights and AdamW optimizer (LR: 2e-5, Weight Decay: 0.01).
- 100% offline, local PyTorch inference with zero external API dependencies.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
MAX_TOKEN_LENGTH = 512
LABEL2ID = {"SAFE": 0, "UNSAFE": 1}
ID2LABEL = {0: "SAFE", 1: "UNSAFE"}


class KidSafeTextDataset(Dataset):
    """PyTorch Dataset for text sequences with fast batch tokenization."""

    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, max_length: int = MAX_TOKEN_LENGTH):
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Loads the pretrained Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def get_model(model_name: str = DEFAULT_MODEL_NAME, num_labels: int = 2):
    """
    Instantiates AutoModelForSequenceClassification backbone with
    dropout and linear sequence classification head.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    # Ensure standard float32 precision for seamless CPU execution
    if not torch.cuda.is_available():
        model = model.float()
    return model


def get_optimizer(model: nn.Module, lr: float = 2e-5, weight_decay: float = 0.01) -> AdamW:
    """Configures AdamW optimizer with standard weight decay and learning rate."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


def get_loss_fn(class_weights: torch.Tensor | None = None) -> nn.CrossEntropyLoss:
    """Returns CrossEntropyLoss with optional balancing class weights."""
    return nn.CrossEntropyLoss(weight=class_weights)


def save_transformer_model(model, tokenizer, output_dir: str):
    """Persists fine-tuned model checkpoint and tokenizer files locally."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer successfully saved to: {output_dir}")


def load_transformer_model(model_dir: str, device: str | None = None):
    """Loads a locally saved fine-tuned Transformer model and tokenizer."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model checkpoint directory not found at: {model_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_sequence(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    threshold: float = 0.35,
    max_length: int = MAX_TOKEN_LENGTH
) -> dict:
    """
    Executes 100% offline local inference on a scene description text.
    Calibrated Decision Engine:
      - If P(UNSAFE) >= threshold (0.35) -> UNSAFE
      - If P(UNSAFE) <  threshold (0.35) -> SAFE
    """
    model.eval()
    inputs = tokenizer(
        str(text),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    prob_safe = float(probabilities[0])
    prob_unsafe = float(probabilities[1])

    verdict = "UNSAFE" if prob_unsafe >= threshold else "SAFE"
    confidence = prob_unsafe if verdict == "UNSAFE" else prob_safe

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "raw_narrative": text,
        "probabilities": {
            "SAFE": round(prob_safe, 4),
            "UNSAFE": round(prob_unsafe, 4)
        },
        "threshold": threshold
    }
