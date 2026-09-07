"""
nlp.py
Classical Natural Language Processing (NLP) module.
Handles text cleaning, token normalization, and TF-IDF n-gram vectorization.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text: str) -> str:
    """
    Classical NLP text preprocessing:
    - Lowercases text
    - Strips URLs and email patterns
    - Removes non-alphanumeric noise while retaining word boundaries
    - Collapses multiple whitespace into single space
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML-like tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation except hyphens inside words
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Replace digits with placeholder or clean spaces
    text = re.sub(r"\d+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_tfidf_vectorizer(
    max_features: int = 25000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Builds a classical TF-IDF Vectorizer configured with:
    - Unigrams and Bigrams (1, 2)
    - Sublinear TF scaling (logarithmic frequency weighting)
    - English stop-words filtering
    - Custom preprocessor
    """
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode"
    )
    return vectorizer
