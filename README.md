# RAG Document Q&A

Retrieval-augmented generation over a small document set, built with
**LangChain** and a **Chroma vector database**: documents are loaded
with LangChain loaders, split with LangChain's
`RecursiveCharacterTextSplitter`, embedded, stored in Chroma, and
retrieved to build a grounded prompt ("answer only from the context")
through a LangChain Runnable (LCEL) chain -- served via a CLI or a
FastAPI endpoint.

## Tech stack
Python · **LangChain** (loaders, text splitters, PromptTemplate, LCEL
chains) · **Chroma** (vector database) · Embeddings (LangChain
`Embeddings` interface: TF-IDF by default, Hugging Face
`all-MiniLM-L6-v2` optional) · FastAPI

### What is used where (all in code, not just mentioned)
| Piece | Implementation |
|---|---|
| Document loading | `langchain_community.document_loaders.DirectoryLoader` / `TextLoader` in `rag_qa/langchain_chain.py` |
| Chunking | `RecursiveCharacterTextSplitter` (800 chars, 120 overlap) |
| Embeddings | `TfidfEmbeddings(Embeddings)` implementing LangChain's interface; `HuggingFaceEmbeddings` when `EMBEDDER=hf` |
| Vector database | **Chroma** via `langchain_chroma.Chroma.from_documents(...)` (FAISS, then LangChain `InMemoryVectorStore`, as fallbacks) |
| Prompt template | `PromptTemplate.from_template(...)` -- the grounded RAG prompt |
| Orchestration | LCEL chain: `RunnableLambda(retrieve) \| RunnableLambda(prompt) \| RunnableLambda(answer)` |

> **Offline fallback:** if LangChain/Chroma cannot be installed (e.g. a
> restricted CI box), `rag_qa/pipeline.py` automatically falls back to
> a dependency-light implementation (own chunker, TF-IDF, cosine
> store, same grounded prompt and answer shape) so the project and its
> core tests still run. With the normal `requirements.txt` install,
> the LangChain + Chroma path is the one that runs -- `GET /health`
> reports `backend`, `store` and `embedder` so you can verify it.
> The default answerer is local/extractive (no API key needed); set
> `OPENAI_API_KEY` to answer with an OpenAI-compatible LLM using the
> same grounded prompt.

## How to run
```bash
pip install -r requirements.txt
python cli.py "What happens when drift is detected?"
uvicorn rag_qa.api:app --reload   # POST /ask, GET /health
pytest                            # run the tests (incl. LangChain/Chroma tests)
```
Sample documents live in `data/sample_docs/` (ML deployment, model
monitoring, the churn pipeline, recommendation features) -- replace
them with your own .md/.txt files to query your own content.

## Project structure
```
rag_qa/langchain_chain.py  PRIMARY: LangChain loaders/splitter, embeddings,
                           Chroma vector DB, PromptTemplate, LCEL RAG chain
rag_qa/pipeline.py         RagPipeline: LangChain path, offline fallback path
rag_qa/chunking.py         fallback chunker (overlapping word chunks)
rag_qa/embeddings.py       fallback TF-IDF embedder + sentence-transformers
rag_qa/store.py            fallback cosine store + optional FAISS store
rag_qa/qa.py               grounded prompt, extractive + LLM answerers
rag_qa/api.py              FastAPI /health and /ask
cli.py                     command-line interface
data/sample_docs/          sample knowledge base
tests/                     chunking, retrieval, prompt, LangChain/Chroma tests
```
