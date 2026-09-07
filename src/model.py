"""
model.py
Classical Machine Learning Classification module.
Constructs a Scikit-Learn Pipeline combining classical NLP (TF-IDF)
with Logistic Regression. Includes helper methods for inspection and persistence.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .nlp import create_tfidf_vectorizer


def build_pipeline(
    max_features: int = 25000,
    ngram_range: tuple[int, int] = (1, 2),
    c_param: float = 1.5,
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end Classical NLP + Logistic Regression Pipeline:
    Step 1: TF-IDF Vectorizer (Unigram + Bigram, Sublinear TF, Preprocessed)
    Step 2: Logistic Regression (L2 regularization, balanced weights, L-BFGS solver)
    """
    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True
    )

    classifier = LogisticRegression(
        C=c_param,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])

    return pipeline


def save_model(pipeline: Pipeline, filepath: str) -> str:
    """Serializes the trained pipeline into a .joblib file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Model saved successfully to: {filepath}")
    return filepath


def load_model(filepath: str) -> Pipeline:
    """Loads a serialized .joblib pipeline."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def extract_top_features(pipeline: Pipeline, top_n: int = 25) -> dict:
    """
    Extracts the most influential words/n-grams learned by the model
    for SAFE (Kid-Friendly) vs UNSAFE (Not Kid-Friendly).
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    # In binary classification, coef_[0] represents the positive class (or UNSAFE if sorted alphabetically)
    # Let's verify class ordering:
    classes = classifier.classes_
    coefficients = classifier.coef_[0]

    # Sort indices by coefficient magnitude
    top_negative_idx = np.argsort(coefficients)[:top_n]
    top_positive_idx = np.argsort(coefficients)[::-1][:top_n]

    return {
        "classes": list(classes),
        "class_0_features": [(feature_names[i], round(float(coefficients[i]), 4)) for i in top_negative_idx],
        "class_1_features": [(feature_names[i], round(float(coefficients[i]), 4)) for i in top_positive_idx]
    }
