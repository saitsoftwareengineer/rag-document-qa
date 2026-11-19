"""FastAPI wrapper: GET /health, POST /ask {"question": "..."}"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_qa.pipeline import RagPipeline

app = FastAPI(title="RAG Document Q&A", version="1.0.0")
_pipeline = RagPipeline()
_pipeline.ingest()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_pipeline.chunks),
            "embedder": _pipeline.embedder.name, "store": _pipeline.store.name}


@app.post("/ask")
def ask(req: AskRequest):
    return _pipeline.ask(req.question, k=req.k)
