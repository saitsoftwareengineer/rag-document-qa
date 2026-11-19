"""Ingest -> chunk -> embed -> index -> retrieve -> answer.

Primary path: the LangChain chain in `rag_qa/langchain_chain.py`
(LangChain loaders + RecursiveCharacterTextSplitter, LangChain
embeddings, Chroma vector database, PromptTemplate-orchestrated
Runnable chain). Fallback path: the dependency-light implementation
below (own chunker + TF-IDF + cosine store), used automatically when
LangChain/Chroma are not installed, so the project always runs.
"""
from __future__ import annotations

import glob
import os

from rag_qa.chunking import Chunk, chunk_text
from rag_qa.embeddings import get_embedder
from rag_qa.qa import build_prompt, get_answerer
from rag_qa.store import get_store

try:
    from rag_qa.langchain_chain import LANGCHAIN_AVAILABLE, LangChainRagChain
except Exception:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False

DEFAULT_DOCS = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")


class RagPipeline:
    def __init__(self, embedder_kind: str = "tfidf", backend: str = "auto"):
        self.backend = (
            "langchain" if (backend == "auto" and LANGCHAIN_AVAILABLE)
            else "fallback"
        )
        self._chain = None
        self.embedder = get_embedder(embedder_kind)
        self.store = get_store()
        self.answerer = get_answerer()
        self.chunks = []

    # ------------------------------------------------------------- ingest
    def ingest(self, docs_dir: str = DEFAULT_DOCS) -> int:
        if self.backend == "langchain":
            self._chain = LangChainRagChain(docs_dir)
            self.chunks = self._chain.chunks
            return len(self.chunks)
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

    # ----------------------------------------------------------- retrieve
    def retrieve(self, question: str, k: int = 3):
        if self.backend == "langchain":
            scored = self._chain.vector_store.similarity_search_with_score(
                question, k=k
            )
            return [
                (Chunk(text=d.page_content,
                       source=os.path.basename(str(d.metadata.get("source", "doc"))),
                       chunk_id=i), float(s))
                for i, (d, s) in enumerate(scored)
            ]
        q_vec = self.embedder.transform([question])[0]
        return self.store.search(q_vec, k=k)

    # ---------------------------------------------------------------- ask
    def ask(self, question: str, k: int = 3) -> dict:
        if self.backend == "langchain":
            return self._chain.ask(question, k=k)
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

    # -------------------------------------------------------------- misc
    @property
    def embedder_name(self) -> str:
        if self.backend == "langchain" and self._chain is not None:
            return self._chain.embedder_name
        return self.embedder.name

    @property
    def store_name(self) -> str:
        if self.backend == "langchain" and self._chain is not None:
            return self._chain.store_name
        return self.store.name
