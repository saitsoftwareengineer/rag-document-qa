"""Vector store: FAISS when available, else an in-code cosine store.

Both expose the same `add(vectors, chunks)` / `search(vector, k)` API,
so the rest of the pipeline does not care which one is active.
"""
from __future__ import annotations

import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


class CosineStore:
    name = "cosine"

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list = []

    def add(self, vectors, chunks):
        self.vectors = _normalise(np.asarray(vectors, dtype=float))
        self.chunks = list(chunks)

    def search(self, query_vector, k: int = 3):
        q = _normalise(np.asarray(query_vector, dtype=float).reshape(1, -1))[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class FaissStore(CosineStore):
    name = "faiss"

    def __init__(self):
        import faiss

        self._faiss = faiss
        self.chunks = []
        self.index = None

    def add(self, vectors, chunks):
        vecs = _normalise(np.asarray(vectors, dtype=float)).astype("float32")
        self.index = self._faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks = list(chunks)

    def search(self, query_vector, k: int = 3):
        q = _normalise(
            np.asarray(query_vector, dtype=float).reshape(1, -1)
        ).astype("float32")
        scores, idx = self.index.search(q, k)
        return [
            (self.chunks[i], float(s))
            for i, s in zip(idx[0], scores[0]) if i >= 0
        ]


def get_store():
    try:
        return FaissStore()
    except Exception:
        return CosineStore()
