"""
sentence_scanner.py
Comprehensive sentence-level safety audit module (Stage 3):
- Analyzes Gemini's multi-sentence video narrative across all scenes.
- Evaluates every sentence thoroughly using the local fine-tuned DeBERTa-v3 Transformer.
- Aggregates full sequence safety verdicts without early-exit interruptions.
- Detects all violating statements and reports comprehensive safety metrics.
"""

import os
import re
import joblib


def split_into_sentences(text: str) -> list[str]:
    """
    Splits multi-line text into individual clean sentences using regex.
    Handles standard sentence endings (.!?), newlines, and bullet points.
    """
    if not text or not isinstance(text, str):
        return []

    # Replace newlines and bullet characters with sentence boundaries
    normalized = re.sub(r"[\r\n]+", ". ", text)
    normalized = re.sub(r"^[•\-\*]\s*", "", normalized)
    normalized = re.sub(r"\s+[•\-\*]\s*", ". ", normalized)

    # Split on sentence boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)

    clean_sentences = []
    for s in raw_sentences:
        clean = s.strip()
        clean = re.sub(r"^[^\w]+", "", clean)
        if len(clean) >= 8:  # ignore trivial fragments
            clean_sentences.append(clean)

    return clean_sentences


class SentenceSafetyScanner:
    """Full-sequence safety scanner wrapping local Transformer model."""

    def __init__(self, model_path: str = None, threshold: float = 0.35):
        self.threshold = threshold
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        deberta_dir = os.path.join(base_dir, "models", "deberta_kid_safe")
        legacy_path = os.path.join(base_dir, "models", "kid_safe_classifier.joblib")

        if model_path is not None and os.path.isdir(model_path):
            deberta_dir = model_path
        elif model_path is not None and os.path.isfile(model_path):
            legacy_path = model_path

        if os.path.exists(deberta_dir):
            from .model import load_transformer_model, predict_sequence
            self.model_type = "transformer"
            self.model, self.tokenizer, self.device = load_transformer_model(deberta_dir)
            self.predict_fn = predict_sequence
        elif os.path.exists(legacy_path):
            self.model_type = "legacy"
            self.model = joblib.load(legacy_path)
            self.classes = list(self.model.classes_)
            self.safe_idx = self.classes.index("SAFE") if "SAFE" in self.classes else 0
            self.unsafe_idx = self.classes.index("UNSAFE") if "UNSAFE" in self.classes else 1
        else:
            raise FileNotFoundError(
                f"No model found at {deberta_dir} or {legacy_path}. Run train.py first."
            )

    def scan_text(self, text: str) -> dict:
        """
        Scans complete text across all sentences and returns a full audit log.
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return {
                "final_decision": "SAFE",
                "is_safe": True,
                "total_sentences": 0,
                "unsafe_count": 0,
                "safe_count": 0,
                "violating_sentences": [],
                "history": []
            }

        history = []
        violating_sentences = []

        for idx, sentence in enumerate(sentences, start=1):
            if self.model_type == "transformer":
                res = self.predict_fn(
                    self.model,
                    self.tokenizer,
                    sentence,
                    device=self.device,
                    threshold=self.threshold
                )
                pred = res["verdict"]
                safe_prob = res["probabilities"]["SAFE"]
                unsafe_prob = res["probabilities"]["UNSAFE"]
                conf = res["confidence"]
            else:
                pred = self.model.predict([sentence])[0]
                probs = self.model.predict_proba([sentence])[0]
                safe_prob = float(probs[self.safe_idx])
                unsafe_prob = float(probs[self.unsafe_idx])
                conf = float(max(probs))

            record = {
                "index": idx,
                "sentence": sentence,
                "prediction": pred,
                "safe_probability": round(safe_prob, 4),
                "unsafe_probability": round(unsafe_prob, 4),
                "confidence": round(conf, 4)
            }
            history.append(record)

            if pred == "UNSAFE":
                violating_sentences.append({
                    "index": idx,
                    "sentence": sentence,
                    "confidence": round(unsafe_prob, 4)
                })

        is_safe = len(violating_sentences) == 0
        final_decision = "SAFE" if is_safe else "UNSAFE"
        avg_safe_conf = sum(h["safe_probability"] for h in history) / len(history)

        return {
            "final_decision": final_decision,
            "is_safe": is_safe,
            "total_sentences": len(sentences),
            "unsafe_count": len(violating_sentences),
            "safe_count": len(sentences) - len(violating_sentences),
            "violating_sentences": violating_sentences,
            "safe_average_confidence": round(avg_safe_conf, 4),
            "history": history
        }
