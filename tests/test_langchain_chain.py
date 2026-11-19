"""Tests for the primary LangChain + Chroma RAG path.

Skipped (not failed) when LangChain is not installed, so the offline
fallback test-suite still passes in minimal environments. In the
project's normal environment these tests run for real.
"""
import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_text_splitters")

from rag_qa.langchain_chain import (  # noqa: E402
    LANGCHAIN_AVAILABLE, RAG_PROMPT, LangChainRagChain, TfidfEmbeddings,
)


def test_langchain_available_and_prompt_template_renders():
    assert LANGCHAIN_AVAILABLE
    rendered = RAG_PROMPT.format(context="CTX", question="Q?")
    assert "CTX" in rendered and "Q?" in rendered
    assert "only the context" in rendered


def test_tfidf_embeddings_follow_langchain_interface():
    emb = TfidfEmbeddings().fit(["cats are animals", "stocks and bonds"])
    doc_vecs = emb.embed_documents(["cats are animals"])
    query_vec = emb.embed_query("cats")
    assert len(doc_vecs) == 1 and len(doc_vecs[0]) == len(query_vec)
    assert max(query_vec) > 0


def test_chroma_chain_retrieves_and_answers(tmp_path):
    chain = LangChainRagChain(
        "data/sample_docs", persist_directory=str(tmp_path / "chroma")
    )
    # The vector database in use is reported honestly (chroma preferred)
    assert chain.store_name in ("chroma", "faiss", "in-memory")
    out = chain.ask("What is tracked for data drift and latency?")
    assert out["answer"]
    assert out["sources"], "chain returned no sources"
    top_source = out["sources"][0]["source"]
    assert top_source in ("monitoring.md", "ml_deployment.md")
    assert "only the context" in out["prompt"]


def test_pipeline_prefers_langchain_backend():
    from rag_qa.pipeline import RagPipeline

    pipe = RagPipeline()
    assert pipe.backend == "langchain"
    pipe.ingest()
    assert pipe.store_name in ("chroma", "faiss", "in-memory")
    out = pipe.ask("How are models deployed in this system?")
    assert out["answer"] and out["sources"]
