"""
predict.py
Inference module for the Kid-Friendly Video Classifier (Stage 3):
- Loads local fine-tuned DeBERTa-v3 Transformer model from models/deberta_kid_safe/
- 100% offline, local PyTorch inference with zero external API dependencies.
- Calibrated Decision Engine: If P(UNSAFE) >= 0.35 -> UNSAFE, else SAFE.
- Outputs structured JSON result with verdict, confidence, and narrative.
"""

import os
import sys
import json
import argparse

# Add repository root to python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import load_transformer_model, predict_sequence

DEFAULT_MODEL_DIR = os.path.join(ROOT_DIR, "models", "deberta_kid_safe")


def classify_description(
    text: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    threshold: float = 0.35,
    device: str | None = None
) -> dict:
    """
    Runs local Transformer inference on input narrative text.
    Returns:
      {
        "verdict": "SAFE" | "UNSAFE",
        "confidence": float,
        "raw_narrative": str,
        "probabilities": {"SAFE": float, "UNSAFE": float},
        "threshold": float
      }
    """
    model, tokenizer, dev = load_transformer_model(model_dir, device=device)
    result = predict_sequence(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=dev,
        threshold=threshold
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Classify video scene description using local DeBERTa-v3")
    parser.add_argument("--text", type=str, help="Video scene description text to evaluate")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to local trained model weights")
    parser.add_argument("--threshold", type=float, default=0.35, help="Decision threshold for UNSAFE classification")
    parser.add_argument("--json", action="store_true", help="Print output in raw JSON format")
    args = parser.parse_args()

    if args.text:
        res = classify_description(args.text, model_dir=args.model_dir, threshold=args.threshold)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\nVerdict: {res['verdict']} (Confidence: {res['confidence']*100:.2f}%)")
            print(f"Probabilities -> SAFE: {res['probabilities']['SAFE']*100:.2f}%, UNSAFE: {res['probabilities']['UNSAFE']*100:.2f}% (Threshold: {res['threshold']})")
            print(f"Narrative: \"{res['raw_narrative']}\"")
    else:
        test_examples = [
            "The video displays a group of children playing with building blocks in a colorful kindergarten classroom while laughing.",
            "The video shows a man holding a handgun pointing it at another person's chest and firing multiple gunshots.",
            "The video depicts a cartoon kitten learning how to paint with watercolors in a friendly animated educational show.",
            "The video captures a violent street fight where a person is stabbed with a knife and bleeding heavily on the ground.",
            "The video displays a nature scene of dolphins swimming gracefully in clear turquoise ocean water.",
            "The video captures an armed confrontation where a masked perpetrator threatens people with a weapon and demands valuables."
        ]

        print("=" * 80)
        print(" Stage 3: Local Transformer Inference Demonstrations (DeBERTa-v3)")
        print(" 100% Offline PyTorch Inference | Zero Third-Party API Calls")
        print("=" * 80)

        for example in test_examples:
            res = classify_description(example, model_dir=args.model_dir, threshold=args.threshold)
            verdict = res["verdict"]
            status = "[  SAFE  ]" if verdict == "SAFE" else "[ UNSAFE ]"
            conf = res["confidence"] * 100
            p_unsafe = res["probabilities"]["UNSAFE"] * 100
            print(f"\nNarrative: \"{example}\"")
            print(f"Decision:  {status} | Confidence: {conf:.2f}% | P(UNSAFE): {p_unsafe:.2f}%")


if __name__ == "__main__":
    main()
