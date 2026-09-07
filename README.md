# Kid-Friendly Video Classification ML Project

A dedicated Machine Learning text classifier designed to determine whether a video's content is **`SAFE`** (Kid-Friendly) or **`UNSAFE`** (Not Kid-Friendly) based on automated video scene summaries (such as those produced by the Gemini API).

---

## Architecture Overview

```
[ 10-Second Video ] ──▶ [ Gemini Multimodal Vision API ] ──▶ [ Text Scene Description ]
                                                                      │
                                                                      ▼
                                                       [ Classical NLP (TF-IDF) ]
                                                                      │
                                                                      ▼
                                                       [ Logistic Regression Classifier ]
                                                                      │
                                                                      ▼
                                                          SAFE   or   UNSAFE
```

- **Input**: 10-second video scene description text.
- **Preprocessing (Classical NLP)**: Text normalization, lowercasing, punctuation stripping, stop-word removal.
- **Feature Extraction**: TF-IDF N-grams (unigrams + bigrams), sublinear term frequency scaling, 25,000 max features.
- **Model**: Regularized Logistic Regression with balanced class weights and probability calibration.
- **Output**: Pure binary decision: **`SAFE`** or **`UNSAFE`** (with confidence probability).

---

## Dataset: SafeWatch-Bench-200K

The model is trained from scratch using a balanced slice of **25,000 samples** from [**`Zhaorun/SafeWatch-Bench-200K-720P`**](https://huggingface.co/datasets/Zhaorun/SafeWatch-Bench-200K-720P):
- **Training Set**: 20,000 samples (80%)
- **Validation Set**: 2,500 samples (10%)
- **Testing Set**: 2,500 samples (10%)

Data is streamed on-the-fly (`streaming=True`) to prevent memory overload and maintain a lightweight disk footprint.

---

## Project Structure

```
Kid-Friendly-Video-Classification-ML-Project/
├── data/
│   ├── safewatch_train.csv         # 20,000 training samples
│   ├── safewatch_val.csv           # 2,500 validation samples
│   ├── safewatch_test.csv          # 2,500 testing samples
│   └── test_predictions.csv       # Test set predictions with probabilities
├── models/
│   └── kid_safe_classifier.joblib  # Serialized trained model (<3 MB)
├── src/
│   ├── dataset.py                  # SafeWatch-Bench streaming & 25k split generator
│   ├── nlp.py                      # Classical text cleaning & TF-IDF vectorizer
│   ├── model.py                    # Scikit-Learn Pipeline & Logistic Regression
│   └── evaluate.py                 # Metric calculations & summary report generator
├── test_results_summary.md         # Detailed testing metrics, confusion matrix & signals
├── train.py                        # Full training & evaluation pipeline runner
├── predict.py                      # Fast inference CLI / helper function
└── README.md                       # Project documentation
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install scikit-learn pandas datasets joblib
```

### 2. Train the Model
```bash
python train.py
```
This automatically:
1. Streams 25,000 balanced video scene descriptions.
2. Trains the Classical NLP + Logistic Regression pipeline in seconds.
3. Evaluates on the test split.
4. Generates `test_results_summary.md` and saves the model to `models/kid_safe_classifier.joblib`.

### 3. Test Any Video Description
```bash
python predict.py --text "A friendly cartoon bear teaching colors to children in a bright classroom"
# Output: Result: SAFE (Confidence: 97.8%)

python predict.py --text "A dark room where a person attacks another person with a knife and blood"
# Output: Result: UNSAFE (Confidence: 98.4%)
```
