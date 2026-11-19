"""Embedders.

Default: a dependency-free TF-IDF embedder, so ingestion, retrieval and
tests run fully offline. Optional: Hugging Face sentence-transformers
(`all-MiniLM-L6-v2` by default) when the package and model are
available -- set EMBEDDER=sentence-transformers to use it.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    name = "tfidf"

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.toarray().astype(float)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(float)


class SentenceTransformerEmbedder:
    """Small HF embedding model; imported lazily, downloads ~90 MB once."""

    name = "sentence-transformers"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts):
        return np.asarray(self.model.encode(texts), dtype=float)

    def transform(self, texts):
        return np.asarray(self.model.encode(texts), dtype=float)


def get_embedder(kind: str = "tfidf"):
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return TfidfEmbedder()
