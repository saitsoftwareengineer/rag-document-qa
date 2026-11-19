"""Retrieval-augmented answering.

Pipeline: embed the question -> retrieve top-k chunks -> build a
grounded prompt (context + question + 'answer only from the context')
-> answer. The default answerer is extractive and local (it returns the
most relevant retrieved sentences), so the system works offline with no
API key. If OPENAI_API_KEY is set, an OpenAI-compatible chat model can
be used instead, with the same grounded prompt.
"""
from __future__ import annotations

import os
import re

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, contexts: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        context="\n\n".join(contexts), question=question
    )


class ExtractiveAnswerer:
    """Returns the retrieved sentences most relevant to the question."""

    name = "extractive"

    def answer(self, question: str, contexts: list[str]) -> str:
        q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
        sentences = []
        for ctx in contexts:
            sentences.extend(re.split(r"(?<=[.!?])\s+", ctx))
        def score(s: str) -> int:
            return len(q_words & set(re.findall(r"[a-z0-9]+", s.lower())))
        best = max(sentences, key=score) if sentences else ""
        return best.strip()


class OpenAICompatibleAnswerer:
    """Optional LLM answerer (OpenAI-compatible chat completions API)."""

    name = "openai-compatible"

    def __init__(self, model: str = "gpt-4o-mini"):
        import httpx

        self._httpx = httpx
        self.model = model
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

    def answer(self, question: str, contexts: list[str]) -> str:
        prompt = build_prompt(question, contexts)
        resp = self._httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_answerer():
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAICompatibleAnswerer()
        except Exception:
            pass
    return ExtractiveAnswerer()
