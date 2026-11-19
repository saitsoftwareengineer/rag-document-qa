"""Split documents into overlapping chunks for embedding / retrieval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def chunk_text(text: str, source: str = "doc", chunk_words: int = 60,
               overlap_words: int = 15) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = " ".join(words[start:start + chunk_words])
        if piece.strip():
            chunks.append(Chunk(text=piece, source=source, chunk_id=i))
        if start + chunk_words >= len(words):
            break
    return chunks
