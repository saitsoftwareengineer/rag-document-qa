"""Ingest -> chunk -> embed -> index -> retrieve -> answer."""
from __future__ import annotations

import glob
import os

from rag_qa.chunking import chunk_text
from rag_qa.embeddings import get_embedder
from rag_qa.qa import build_prompt, get_answerer
from rag_qa.store import get_store

DEFAULT_DOCS = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")


class RagPipeline:
    def __init__(self, embedder_kind: str = "tfidf"):
        self.embedder = get_embedder(embedder_kind)
        self.store = get_store()
        self.answerer = get_answerer()
        self.chunks = []

    def ingest(self, docs_dir: str = DEFAULT_DOCS) -> int:
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(docs_dir, "*"))):
            with open(path) as f:
                text = f.read()
            self.chunks.extend(
                chunk_text(text, source=os.path.basename(path))
            )
        vectors = self.embedder.fit_transform([c.text for c in self.chunks])
        self.store.add(vectors, self.chunks)
        return len(self.chunks)

    def retrieve(self, question: str, k: int = 3):
        q_vec = self.embedder.transform([question])[0]
        return self.store.search(q_vec, k=k)

    def ask(self, question: str, k: int = 3) -> dict:
        results = self.retrieve(question, k=k)
        contexts = [chunk.text for chunk, _ in results]
        return {
            "question": question,
            "answer": self.answerer.answer(question, contexts),
            "prompt": build_prompt(question, contexts),
            "sources": [
                {"source": c.source, "score": round(s, 4)} for c, s in results
            ],
        }
