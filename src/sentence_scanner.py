"""
sentence_scanner.py
Implements sentence-by-sentence early-exit safety scanning.
Analyzes Gemini's multi-sentence video summary sentence by sentence:
- Checks each sentence individually with our custom trained classifier.
- If even ONE sentence is classified as UNSAFE, execution halts immediately
  (early exit without reading the rest of the text) and outputs UNSAFE.
- If all sentences pass inspection, it outputs SAFE.
"""

import os
import re
import joblib


def split_into_sentences(text: str) -> list[str]:
    """
    Splits vast multi-line text into individual clean sentences using regex.
    Handles standard sentence endings (.!?), newlines, and bullet points.
    """
    if not text or not isinstance(text, str):
        return []

    # Replace newlines and bullet characters with sentence boundaries
    normalized = re.sub(r"[\r\n]+", ". ", text)
    normalized = re.sub(r"^[•\-\*]\s*", "", normalized)
    normalized = re.sub(r"\s+[•\-\*]\s*", ". ", normalized)

    # Split on sentence boundaries: period, question mark, exclamation mark followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)

    clean_sentences = []
    for s in raw_sentences:
        clean = s.strip()
        # Remove trailing/leading punctuation artifacts and keep meaningful sentences
        clean = re.sub(r"^[^\w]+", "", clean)
        if len(clean) >= 8:  # ignore trivial fragments
            clean_sentences.append(clean)

    return clean_sentences


class SentenceSafetyScanner:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to the trained model in models/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "kid_safe_classifier.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}. Run train.py first.")

        self.model = joblib.load(model_path)
        self.classes = list(self.model.classes_)
        self.safe_idx = self.classes.index("SAFE") if "SAFE" in self.classes else 0
        self.unsafe_idx = self.classes.index("UNSAFE") if "UNSAFE" in self.classes else 1

    def scan_text(self, text: str) -> dict:
        """
        Scans Gemini's vast text output sentence by sentence with early exit.
        
        Returns:
            dict containing:
            - final_decision: 'SAFE' or 'UNSAFE'
            - is_safe: bool
            - early_exit_triggered: bool
            - trigger_sentence: str (the unsafe sentence that stopped execution, or None)
            - trigger_index: int (1-based index where scanning stopped, or None)
            - total_sentences: int (total number of sentences in input text)
            - sentences_scanned_count: int (number of sentences actually evaluated before stopping)
            - history: list of evaluated sentences with their individual predictions & confidence
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return {
                "final_decision": "SAFE",
                "is_safe": True,
                "early_exit_triggered": False,
                "trigger_sentence": None,
                "trigger_index": None,
                "trigger_confidence": 0.0,
                "total_sentences": 0,
                "sentences_scanned_count": 0,
                "history": []
            }

        history = []

        for idx, sentence in enumerate(sentences, start=1):
            pred = self.model.predict([sentence])[0]
            probs = self.model.predict_proba([sentence])[0]

            safe_prob = float(probs[self.safe_idx])
            unsafe_prob = float(probs[self.unsafe_idx])

            record = {
                "index": idx,
                "sentence": sentence,
                "prediction": pred,
                "safe_probability": round(safe_prob, 4),
                "unsafe_probability": round(unsafe_prob, 4),
                "confidence": round(float(max(probs)), 4)
            }
            history.append(record)

            # CRITICAL REQUIREMENT:
            # If ONE unsafe sentence is detected -> IMMEDIATELY STOP EXECUTION!
            # Does not read the rest of the vast text.
            if pred == "UNSAFE":
                return {
                    "final_decision": "UNSAFE",
                    "is_safe": False,
                    "early_exit_triggered": True,
                    "trigger_sentence": sentence,
                    "trigger_index": idx,
                    "trigger_confidence": round(unsafe_prob, 4),
                    "total_sentences": len(sentences),
                    "sentences_scanned_count": idx,
                    "history": history
                }

        # If loop completes without any unsafe sentence:
        avg_safe_conf = sum(h["safe_probability"] for h in history) / len(history)

        return {
            "final_decision": "SAFE",
            "is_safe": True,
            "early_exit_triggered": False,
            "trigger_sentence": None,
            "trigger_index": None,
            "trigger_confidence": 0.0,
            "total_sentences": len(sentences),
            "sentences_scanned_count": len(sentences),
            "safe_average_confidence": round(avg_safe_conf, 4),
            "history": history
        }
