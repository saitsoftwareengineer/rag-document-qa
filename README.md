# RAG Document Q&A

Retrieval-augmented generation over a small document set: ingest
documents, split them into overlapping chunks, embed them, retrieve the
top-k chunks for a question, build a grounded prompt ("answer only from
the context"), and answer -- through a CLI or a FastAPI endpoint.

## Tech stack
Python · Hugging Face Transformers (sentence embeddings,
`all-MiniLM-L6-v2`) · FAISS (optional) · FastAPI · Scikit-learn TF-IDF
(offline fallback embedder)

> **Offline-first design:** the default path uses a TF-IDF embedder and
> an in-code cosine-similarity store, plus an extractive answerer, so
> the whole pipeline and its tests run with no downloads and no API
> key. Set `EMBEDDER=sentence-transformers` (and install
> sentence-transformers) for HF embeddings, install `faiss-cpu` for a
> FAISS index, and set `OPENAI_API_KEY` to answer with an
> OpenAI-compatible LLM instead of the extractive fallback -- the
> grounded prompt is identical in every configuration.

## How to run
```bash
pip install -r requirements.txt
python cli.py "What happens when drift is detected?"
uvicorn rag_qa.api:app --reload   # POST /ask, GET /health
pytest                            # run the tests
```
Sample documents live in `data/sample_docs/` (ML deployment, model
monitoring, the churn pipeline, recommendation features) -- replace
them with your own .md/.txt files to query your own content.

## Project structure
```
rag_qa/chunking.py    overlapping text chunking
rag_qa/embeddings.py  TF-IDF embedder + HF sentence-transformers embedder
rag_qa/store.py       cosine vector store + optional FAISS store
rag_qa/qa.py          grounded prompt, extractive + LLM answerers
rag_qa/pipeline.py    ingest -> retrieve -> answer orchestration
rag_qa/api.py         FastAPI /health and /ask
cli.py                command-line interface
data/sample_docs/     sample knowledge base
tests/                chunking, retrieval, prompt and end-to-end tests
```
