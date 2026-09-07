"""
predict.py
Simple inference utility for the Kid-Friendly Video Classification Model.
Loads the trained model and classifies any video scene description into SAFE or UNSAFE.
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model


def classify_description(text: str, model_path: str = None) -> dict:
    """
    Classifies a video description into SAFE or UNSAFE with confidence probability.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "kid_safe_classifier.joblib")

    model = load_model(model_path)
    classes = list(model.classes_)
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    safe_idx = classes.index("SAFE") if "SAFE" in classes else 0
    unsafe_idx = classes.index("UNSAFE") if "UNSAFE" in classes else 1

    safe_score = float(probabilities[safe_idx])
    unsafe_score = float(probabilities[unsafe_idx])

    return {
        "text": text,
        "prediction": prediction,
        "is_safe": prediction == "SAFE",
        "safe_probability": round(safe_score, 4),
        "unsafe_probability": round(unsafe_score, 4),
        "confidence": round(float(max(probabilities)), 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Classify video description as SAFE or UNSAFE")
    parser.add_argument("--text", type=str, help="Video scene description text to evaluate")
    args = parser.parse_args()

    if args.text:
        res = classify_description(args.text)
        print(f"\nResult: {res['prediction']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Safe Probability: {res['safe_probability']*100:.1f}% | Unsafe Probability: {res['unsafe_probability']*100:.1f}%")
    else:
        test_examples = [
            "The video displays a group of children playing with building blocks in a colorful kindergarten classroom while laughing.",
            "The video shows a man holding a handgun pointing it at another person's chest and firing multiple gunshots.",
            "The video depicts a cartoon kitten learning how to paint with watercolors in a friendly animated educational show.",
            "The video captures a violent street fight where a person is stabbed with a knife and bleeding heavily on the ground.",
            "The video displays a nature scene of dolphins swimming gracefully in clear turquoise ocean water.",
            "The video captures an armed confrontation where a masked perpetrator threatens people with a weapon and demands valuables."
        ]

        print("=" * 70)
        print(" Kid-Friendly Video Classifier: Inference Demonstrations")
        print("=" * 70)
        for example in test_examples:
            res = classify_description(example)
            verdict = res["prediction"]
            prob = res["safe_probability"] if verdict == "SAFE" else res["unsafe_probability"]
            status = "[  SAFE  ]" if verdict == "SAFE" else "[ UNSAFE ]"
            print(f"\nInput: \"{example}\"")
            print(f"Decision: {status} (Confidence: {prob*100:.1f}%)")


if __name__ == "__main__":
    main()
