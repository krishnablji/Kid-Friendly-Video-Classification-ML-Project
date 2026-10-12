"""
dataset.py
Public dataset extraction and cleaning module:
- Streams 25,000 balanced samples from SafeWatch-Bench-200K-720P.
- Filters out adult/sexual content.
- Extracts non-sexual UNSAFE (Violence, Harassment, Hazards) and SAFE samples.
- Creates stratified Train (80%), Validation (10%), and Test (10%) splits.
"""

import os
import json
import re
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

DATASET_NAME = "Zhaorun/SafeWatch-Bench-200K-720P"
TOTAL_SAMPLES = 25000
SAMPLES_PER_CLASS = TOTAL_SAMPLES // 2

EXCLUDED_PATTERNS = {
    "sex", "sexual", "nude", "nudity", "naked", "breast", "breasts", "buttock", "buttocks",
    "lingerie", "underwear", "bra", "bikini", "thong", "penis", "vagina", "genital",
    "genitals", "erotic", "porn", "pornography", "kissing", "seductive", "intercourse",
    "stripper", "cleavage", "masturbat"
}


def is_excluded_content(text: str) -> bool:
    """Checks if text contains excluded adult or sexual keywords."""
    if not text:
        return False
    text_lower = text.lower()
    for term in EXCLUDED_PATTERNS:
        if re.search(r"\b" + re.escape(term), text_lower):
            return True
    return False


def extract_description(response: str) -> str | None:
    """Extracts raw scene description from benchmark response payload."""
    if not response:
        return None
    match = re.search(
        r"DESCRIPTION:\s*(.*?)(?=\s*GUARDRAIL:|\s*EXPLANATION:|$)",
        str(response),
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        desc = match.group(1).strip()
        if len(desc) >= 15:
            return desc
    return None


def parse_guardrail_status(response: str) -> tuple[bool, bool]:
    """Parses structural guardrail flags from dataset JSON."""
    if not response:
        return False, False
    match = re.search(r"GUARDRAIL:\s*(\{.*?\})", str(response), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return False, False
    try:
        data = json.loads(match.group(1))
        has_flag = False
        has_non_flag = False
        for key, val in data.items():
            is_flagged = (val is True or val == 1 or str(val).lower() == "true")
            if "sexual" in key.lower() and is_flagged:
                has_flag = True
            elif is_flagged:
                has_non_flag = True
        return has_flag, has_non_flag
    except Exception:
        return False, False


def collect_dataset(target_total: int = TOTAL_SAMPLES, output_dir: str = "data") -> tuple[str, str, str]:
    """Streams, cleans, and splits the SafeWatch-Bench dataset."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "safewatch_train.csv")
    val_path = os.path.join(output_dir, "safewatch_val.csv")
    test_path = os.path.join(output_dir, "safewatch_test.csv")

    from datasets import load_dataset

    safe_samples = []
    unsafe_samples = []
    seen_descriptions = set()

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    for row in dataset:
        response = row.get("response", "")
        desc = extract_description(response)
        if not desc or desc in seen_descriptions:
            continue

        has_excluded_flag, has_non_sexual_violation = parse_guardrail_status(response)
        if has_excluded_flag or is_excluded_content(desc):
            continue

        seen_descriptions.add(desc)

        if has_non_sexual_violation and len(unsafe_samples) < SAMPLES_PER_CLASS:
            unsafe_samples.append({"description": desc, "label": "UNSAFE"})
        elif (not has_non_sexual_violation) and len(safe_samples) < SAMPLES_PER_CLASS:
            safe_samples.append({"description": desc, "label": "SAFE"})

        if len(safe_samples) >= SAMPLES_PER_CLASS and len(unsafe_samples) >= SAMPLES_PER_CLASS:
            break

    df = pd.DataFrame(safe_samples + unsafe_samples)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


if __name__ == "__main__":
    collect_dataset()
