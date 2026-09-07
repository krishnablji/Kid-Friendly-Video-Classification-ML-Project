"""
train.py
Main training script for the Kid-Friendly Video Classification Model.
Workflow:
1. Loads/extracts 25,000 balanced samples (20k Train, 2.5k Val, 2.5k Test).
2. Builds Classical NLP (TF-IDF unigrams & bigrams) + Logistic Regression pipeline.
3. Fits the model on the 20,000 training samples.
4. Validates on the 2,500 validation samples.
5. Evaluates on the 2,500 test samples.
6. Saves model to 'models/kid_safe_classifier.joblib'.
7. Exports comprehensive test summary report to 'test_results_summary.md'.
"""

import os
import sys
import time
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import collect_dataset
from src.model import build_pipeline, save_model
from src.evaluate import evaluate_dataset, generate_summary_report


def main():
    print("=" * 70)
    print(" Kid-Friendly Video Classification ML Project: Model Training")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Ensure dataset splits exist
    print("\n[Step 1/5] Checking/Extracting 25,000 samples from SafeWatch-Bench...")
    train_path, val_path, test_path = collect_dataset(
        target_total=25000,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    )

    print("\n[Step 2/5] Loading data splits into memory...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"  Training set:   {len(train_df):,} rows")
    print(f"  Validation set: {len(val_df):,} rows")
    print(f"  Testing set:    {len(test_df):,} rows")
    print(f"  Class balance (Train):\n{train_df['label'].value_counts().to_string()}")

    # Step 2: Build Classical NLP + Logistic Regression Pipeline
    print("\n[Step 3/5] Initializing Classical NLP (TF-IDF) + Logistic Regression Pipeline...")
    pipeline = build_pipeline(
        max_features=25000,
        ngram_range=(1, 2),
        c_param=1.5,
        max_iter=1000
    )

    # Step 3: Train Model
    print("\n[Step 4/5] Training model on 20,000 samples...")
    t_train_start = time.time()
    pipeline.fit(train_df["description"].tolist(), train_df["label"].tolist())
    train_duration = time.time() - t_train_start
    print(f"  Training completed in {train_duration:.2f} seconds!")

    # Step 4: Validate
    print("\n[Step 5/5] Evaluating performance...")
    val_metrics = evaluate_dataset(pipeline, val_df)
    print(f"  Validation Accuracy: {val_metrics['accuracy'] * 100:.2f}% | F1: {val_metrics['macro_f1'] * 100:.2f}%")

    test_metrics = evaluate_dataset(pipeline, test_df)
    print(f"  Testing Accuracy:    {test_metrics['accuracy'] * 100:.2f}% | F1: {test_metrics['macro_f1'] * 100:.2f}%")
    print(f"  Confusion Matrix (Actual rows x Predicted cols):\n{test_metrics['confusion_matrix']}")

    # Save Model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_filepath = os.path.join(models_dir, "kid_safe_classifier.joblib")
    save_model(pipeline, model_filepath)

    # Export Test Summary Report
    summary_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_results_summary.md"
    )
    generate_summary_report(pipeline, test_df, test_metrics, summary_filepath)

    # Also save test predictions CSV
    test_pred_df = test_df.copy()
    test_pred_df["predicted_label"] = test_metrics["predictions"]
    test_pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_predictions.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved test predictions to: {test_pred_path}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" ALL TASKS FINISHED SUCCESSFULLY in {total_time:.2f}s!")
    print(f" Saved Model:    {model_filepath}")
    print(f" Test Summary:   {summary_filepath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
