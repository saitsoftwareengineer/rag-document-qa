from rag_qa.chunking import chunk_text
from rag_qa.pipeline import RagPipeline
from rag_qa.qa import build_prompt


def test_chunking_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, source="t", chunk_words=60, overlap_words=15)
    assert len(chunks) >= 3
    assert all(c.text for c in chunks)
    # overlap: end of chunk 0 reappears at the start of chunk 1
    assert chunks[0].text.split()[-1] in chunks[1].text.split()[:20]


def test_retrieval_finds_the_right_document():
    pipe = RagPipeline()
    pipe.ingest()
    results = pipe.retrieve("What is tracked for data drift and latency?", k=2)
    assert results, "no results"
    assert results[0][0].source in ("monitoring.md", "ml_deployment.md")
    assert results[0][1] > 0


def test_grounded_prompt_contains_context_and_question():
    prompt = build_prompt("Why?", ["Some context sentence."])
    assert "Some context sentence." in prompt and "Why?" in prompt
    assert "only the context" in prompt


def test_ask_returns_grounded_answer_with_sources():
    pipe = RagPipeline()
    pipe.ingest()
    out = pipe.ask("How are models deployed in this system?")
    assert out["answer"]
    assert out["sources"] and out["sources"][0]["source"]
    assert "FastAPI" in out["prompt"] or "fastapi" in out["prompt"].lower()
