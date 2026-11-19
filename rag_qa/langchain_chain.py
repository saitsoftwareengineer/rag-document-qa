"""Primary RAG path, built on LangChain.

This module genuinely uses, in code:

* **LangChain document loading** -- `DirectoryLoader` / `TextLoader`
  (langchain-community) load the sample docs as LangChain `Document`s.
* **LangChain chunking** -- `RecursiveCharacterTextSplitter`
  (langchain-text-splitters) splits them with overlap.
* **Embeddings** -- classes implementing LangChain's `Embeddings`
  interface (langchain-core): a TF-IDF embedder that runs fully
  offline, and Hugging Face `HuggingFaceEmbeddings`
  (all-MiniLM-L6-v2) when sentence-transformers is installed.
* **A vector database** -- **Chroma** (langchain-chroma, embedded /
  persistent) as the primary store, with LangChain's
  `InMemoryVectorStore` as the fallback if Chroma is unavailable.
  FAISS is also supported via langchain-community when installed.
* **Prompt template / orchestration** -- a LangChain `PromptTemplate`
  renders the grounded prompt, and the retrieve -> prompt -> answer
  flow is composed as a LangChain Runnable (LCEL) chain.

The dependency-light pipeline in `rag_qa/pipeline.py` remains as an
offline fallback for environments where LangChain/Chroma cannot be
installed; `RagPipeline` prefers this LangChain path when available.
"""
from __future__ import annotations

import os

LANGCHAIN_AVAILABLE = True
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.vectorstores import InMemoryVectorStore
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # older langchain layout
        from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised only without langchain
    LANGCHAIN_AVAILABLE = False

from rag_qa.qa import get_answerer

# ---------------------------------------------------------------- prompt
RAG_PROMPT = PromptTemplate.from_template(
    "Answer the question using only the context below.\n"
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
) if LANGCHAIN_AVAILABLE else None


# ------------------------------------------------------------- embeddings
if LANGCHAIN_AVAILABLE:

    class TfidfEmbeddings(Embeddings):
        """LangChain `Embeddings` implementation backed by TF-IDF.

        Runs fully offline and needs no downloads, which is why it is
        the default in tests/CI. It follows the exact interface Chroma
        and the other LangChain vector stores expect.
        """

        name = "langchain-tfidf"

        def __init__(self):
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._fitted = False

        def fit(self, texts: list[str]):
            self._vectorizer.fit(texts)
            self._fitted = True
            return self

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if not self._fitted:
                self.fit(texts)
            return self._vectorizer.transform(texts).toarray().tolist()

        def embed_query(self, text: str) -> list[float]:
            return self._vectorizer.transform([text]).toarray()[0].tolist()


def get_langchain_embeddings(kind: str | None = None):
    """HF sentence embeddings when available, else the TF-IDF embedder."""
    kind = kind or os.environ.get("EMBEDDER", "tfidf")
    if kind in ("hf", "sentence-transformers", "huggingface"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings

                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception:
                pass  # fall through to the offline embedder
    return TfidfEmbeddings()


# ------------------------------------------------------------ vector store
def build_vector_store(documents, embeddings, persist_directory: str | None = None):
    """Chroma first, then FAISS, then LangChain's in-memory store."""
    try:
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="rag-documents",
            persist_directory=persist_directory,
        )
        store.store_name = "chroma"
        return store
    except Exception:
        pass
    try:
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(documents, embeddings)
        store.store_name = "faiss"
        return store
    except Exception:
        pass
    store = InMemoryVectorStore.from_documents(documents, embeddings)
    store.store_name = "in-memory"
    return store


# ---------------------------------------------------------------- loading
def load_documents(docs_dir: str):
    """Load .md/.txt files as LangChain Documents (DirectoryLoader)."""
    try:
        from langchain_community.document_loaders import (
            DirectoryLoader, TextLoader,
        )

        docs = []
        for glob_pat in ("*.md", "*.txt"):
            loader = DirectoryLoader(
                docs_dir, glob=glob_pat, loader_cls=TextLoader,
                show_progress=False,
            )
            docs.extend(loader.load())
        if docs:
            return docs
    except Exception:
        pass  # fall back to manual loading below
    docs = []
    for name in sorted(os.listdir(docs_dir)):
        if name.endswith((".md", ".txt")):
            with open(os.path.join(docs_dir, name)) as f:
                docs.append(Document(
                    page_content=f.read(), metadata={"source": name}
                ))
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _source_name(doc) -> str:
    source = doc.metadata.get("source", "doc")
    return os.path.basename(str(source))


# ------------------------------------------------------------------ chain
class LangChainRagChain:
    """RAG chain: Chroma retrieval -> PromptTemplate -> answerer.

    The flow is composed as a LangChain Runnable (LCEL) chain:
        RunnableLambda(retrieve) | RunnableLambda(prompt) | RunnableLambda(answer)
    """

    name = "langchain"

    def __init__(self, docs_dir: str, persist_directory: str | None = None,
                 embeddings=None):
        self.documents = load_documents(docs_dir)
        self.chunks = split_documents(self.documents)
        self.embeddings = embeddings or get_langchain_embeddings()
        if isinstance(self.embeddings, TfidfEmbeddings):
            # TF-IDF must see the corpus before Chroma embeds with it
            self.embeddings.fit([c.page_content for c in self.chunks])
        self.vector_store = build_vector_store(
            self.chunks, self.embeddings,
            persist_directory=persist_directory,
        )
        self.store_name = getattr(self.vector_store, "store_name", "unknown")
        self.embedder_name = getattr(self.embeddings, "name", "huggingface")
        self.answerer = get_answerer()
        self.chain = (
            RunnableLambda(self._retrieve_step)
            | RunnableLambda(self._prompt_step)
            | RunnableLambda(self._answer_step)
        )

    # -- chain steps ----------------------------------------------------
    def _retrieve_step(self, payload: dict) -> dict:
        question, k = payload["question"], payload.get("k", 3)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        # similarity scores for source reporting (Chroma/FAISS stores)
        try:
            scored = self.vector_store.similarity_search_with_score(question, k=k)
            score_by_text = {d.page_content: float(s) for d, s in scored}
        except Exception:
            score_by_text = {}
        return {"question": question, "docs": docs, "scores": score_by_text}

    def _prompt_step(self, payload: dict) -> dict:
        context = "\n\n".join(d.page_content for d in payload["docs"])
        payload["prompt"] = RAG_PROMPT.format(context=context,
                                              question=payload["question"])
        return payload

    @staticmethod
    def _to_similarity(store_name: str, raw: float) -> float:
        # Chroma's default space returns L2 *distances* (lower = closer);
        # convert to a 0..1 similarity so reported scores always mean
        # "higher = more relevant", like the cosine fallback store.
        return 1.0 / (1.0 + raw) if store_name == "chroma" else raw

    def _answer_step(self, payload: dict) -> dict:
        contexts = [d.page_content for d in payload["docs"]]
        return {
            "question": payload["question"],
            "answer": self.answerer.answer(payload["question"], contexts),
            "prompt": payload["prompt"],
            "sources": [
                {"source": _source_name(d),
                 "score": round(self._to_similarity(
                     self.store_name,
                     payload["scores"].get(d.page_content, 0.0)), 4)}
                for d in payload["docs"]
            ],
        }

    # -- public API (same shape as the fallback pipeline) ---------------
    def ask(self, question: str, k: int = 3) -> dict:
        return self.chain.invoke({"question": question, "k": k})

    def retrieve(self, question: str, k: int = 3):
        docs = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        ).invoke(question)
        return [(d.page_content, _source_name(d)) for d in docs]
