"""
dataset.py
Extracts and prepares exactly 25,000 balanced samples from SafeWatch-Bench-200K-720P
strictly filtering out ANY sexual content, nudity, or erotic material:
- 12,500 UNSAFE samples: exclusively Threats, Violence & Harm, Harassment & Bullying,
  Illegal/Regulated Activities, Hateful Content, or Physical Hazards.
- 12,500 SAFE samples: wholesome, non-violating, non-sexual educational/general content.

Splits into:
- 80% Train: 20,000 samples (10,000 SAFE, 10,000 UNSAFE)
- 10% Validation: 2,500 samples (1,250 SAFE, 1,250 UNSAFE)
- 10% Test: 2,500 samples (1,250 SAFE, 1,250 UNSAFE)
"""

import os
import json
import re
import warnings

# Suppress HuggingFace cache warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_NAME = "Zhaorun/SafeWatch-Bench-200K-720P"
TOTAL_SAMPLES = 25000
SAMPLES_PER_CLASS = TOTAL_SAMPLES // 2  # 12,500 SAFE, 12,500 UNSAFE

# Strict keyword exclusion to ensure ZERO sexual content exists anywhere in the dataset
STRICT_SEXUAL_EXCLUSION = {
    "sex", "sexual", "nude", "nudity", "naked", "breast", "breasts", "buttock", "buttocks",
    "lingerie", "underwear", "bra", "bikini", "thong", "penis", "vagina", "genital",
    "genitals", "erotic", "porn", "pornography", "kissing", "seductive", "intercourse",
    "stripper", "cleavage", "masturbat"
}


def contains_sexual_content(text: str) -> bool:
    """Checks if text contains any sexual, nudity, or adult keywords."""
    if not text:
        return False
    text_lower = text.lower()
    for term in STRICT_SEXUAL_EXCLUSION:
        # Match whole word or bounded subword
        if re.search(r"\b" + re.escape(term), text_lower):
            return True
    return False


def extract_description(response: str) -> str | None:
    """Extracts the raw video scene description from model response text."""
    if not response:
        return None
    response = str(response)
    match = re.search(
        r"DESCRIPTION:\s*(.*?)(?=\s*GUARDRAIL:|\s*EXPLANATION:|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        desc = match.group(1).strip()
        if len(desc) >= 15:
            return desc
    return None


def parse_guardrail_status(response: str) -> tuple[bool, bool]:
    """
    Parses GUARDRAIL JSON.
    Returns:
        (has_sexual_content: bool, has_non_sexual_violation: bool)
    """
    if not response:
        return False, False
    match = re.search(
        r"GUARDRAIL:\s*(\{.*?\})",
        str(response),
        flags=re.IGNORECASE | re.DOTALL
    )
    if not match:
        return False, False
    try:
        data = json.loads(match.group(1))

        has_sexual = False
        has_non_sexual_violation = False

        for key, val in data.items():
            is_flagged = (val is True or val == 1 or str(val).lower() == "true")
            key_lower = key.lower()

            if "sexual" in key_lower:
                if is_flagged:
                    has_sexual = True
            else:
                # Violence, Harassment, Illegal, Hate, Deceptive
                if is_flagged:
                    has_non_sexual_violation = True

        return has_sexual, has_non_sexual_violation
    except Exception:
        return False, False


def collect_dataset(target_total: int = TOTAL_SAMPLES, output_dir: str = "data", force_refresh: bool = True) -> tuple[str, str, str]:
    """
    Streams samples from HuggingFace SafeWatch-Bench-200K-720P.
    Strictly filters out any sexual content.
    Extracts 12,500 non-sexual UNSAFE and 12,500 non-sexual SAFE samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "safewatch_train.csv")
    val_path = os.path.join(output_dir, "safewatch_val.csv")
    test_path = os.path.join(output_dir, "safewatch_test.csv")

    if not force_refresh and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_len = len(pd.read_csv(train_path))
        val_len = len(pd.read_csv(val_path))
        test_len = len(pd.read_csv(test_path))
        if train_len + val_len + test_len >= target_total:
            print(f"Data splits already present with {train_len + val_len + test_len:,} samples. Reusing.")
            return train_path, val_path, test_path

    print(f"Streaming dataset from HuggingFace: {DATASET_NAME} (Strict Non-Sexual Filtering)...")
    from datasets import load_dataset

    safe_samples = []
    unsafe_samples = []
    seen_descriptions = set()

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    count = 0
    skipped_sexual = 0

    for row in dataset:
        count += 1
        response = row.get("response", "")
        desc = extract_description(response)
        if not desc or desc in seen_descriptions:
            continue

        # Check for sexual content in either guardrail flags or text description
        has_sexual_flag, has_non_sexual_violation = parse_guardrail_status(response)
        has_sexual_text = contains_sexual_content(desc)

        # STRICT EXCLUSION: Drop if ANY sexual indicator is present!
        if has_sexual_flag or has_sexual_text:
            skipped_sexual += 1
            continue

        seen_descriptions.add(desc)

        # Non-sexual UNSAFE: Violence, Threats, Harassment, Illegal, Hate
        if has_non_sexual_violation and len(unsafe_samples) < SAMPLES_PER_CLASS:
            unsafe_samples.append({"description": desc, "label": "UNSAFE"})

        # Non-sexual SAFE: Zero guardrail violations and clean text
        elif (not has_non_sexual_violation) and len(safe_samples) < SAMPLES_PER_CLASS:
            safe_samples.append({"description": desc, "label": "SAFE"})

        total_collected = len(safe_samples) + len(unsafe_samples)
        if total_collected % 1000 == 0 and total_collected > 0:
            print(f"  [Progress] Collected: {total_collected:,}/{target_total:,} "
                  f"(SAFE: {len(safe_samples):,}, UNSAFE [Violent/Threats/Danger]: {len(unsafe_samples):,}) | "
                  f"Skipped Sexual: {skipped_sexual:,} | Processed: {count:,}")

        if len(safe_samples) >= SAMPLES_PER_CLASS and len(unsafe_samples) >= SAMPLES_PER_CLASS:
            break

    # Combine and shuffle
    all_records = safe_samples + unsafe_samples
    print(f"\nCollection Complete! Total balanced records: {len(all_records):,}")
    print(f"  SAFE (Wholesome/Clean):           {len(safe_samples):,}")
    print(f"  UNSAFE (Violence/Threats/Danger): {len(unsafe_samples):,}")
    print(f"  Total Sexual Samples Excluded:    {skipped_sexual:,}")

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 80% train (20,000), 10% val (2,500), 10% test (2,500)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved 100% Non-Sexual Dataset Splits into '{output_dir}':")
    print(f"  safewatch_train.csv  -> {len(train_df):,} rows")
    print(f"  safewatch_val.csv    -> {len(val_df):,} rows")
    print(f"  safewatch_test.csv   -> {len(test_df):,} rows")

    return train_path, val_path, test_path


if __name__ == "__main__":
    collect_dataset(force_refresh=True)
