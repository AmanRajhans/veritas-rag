"""Grounded answer synthesis with citation validation and confidence gating."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from veritas_rag.models import Citation, RetrievedChunk
from veritas_rag.text import informative_tokens, jaccard, sentences, tokenize

CITATION_PATTERN = re.compile(r"\[(\d+)]")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    citation_ids: list[int]


class AnswerGenerator(Protocol):
    def generate(self, query: str, evidence: list[RetrievedChunk]) -> GeneratedAnswer: ...


class ExtractiveGenerator:
    """A dependency-free baseline that can only say things found in evidence."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, query: str, evidence: list[RetrievedChunk]) -> GeneratedAnswer:
        query_tokens = set(informative_tokens(query))
        candidates: list[tuple[float, int, str, set[str]]] = []
        for citation_id, item in enumerate(evidence, start=1):
            for sentence in sentences(item.chunk.text):
                tokens = set(tokenize(sentence))
                if len(tokens) < 4:
                    continue
                # A retrieval rank alone must never make an unrelated sentence answerable.
                if not tokens & query_tokens:
                    continue
                coverage = len(tokens & query_tokens) / max(len(query_tokens), 1)
                density = len(tokens & query_tokens) / max(len(tokens), 1)
                score = 0.55 * coverage + 0.25 * item.score + 0.20 * min(density * 4, 1.0)
                candidates.append((score, citation_id, sentence, tokens))

        chosen: list[tuple[int, str, set[str]]] = []
        for _, citation_id, sentence, tokens in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            if any(jaccard(tokens, selected_tokens) > 0.72 for _, _, selected_tokens in chosen):
                continue
            chosen.append((citation_id, sentence, tokens))
            if len(chosen) >= self.max_sentences:
                break
        if not chosen:
            return GeneratedAnswer(text="", citation_ids=[])
        text = " ".join(f"{sentence} [{citation_id}]" for citation_id, sentence, _ in chosen)
        return GeneratedAnswer(text=text, citation_ids=[item[0] for item in chosen])


class OpenAICompatibleGenerator:
    """Optional adapter for any provider implementing the chat-completions contract."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, query: str, evidence: list[RetrievedChunk]) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{position}] {item.chunk.title} / {item.chunk.section}\n{item.chunk.text}"
            for position, item in enumerate(evidence, start=1)
        )
        prompt = (
            "Answer using only the numbered evidence. Cite every factual sentence with [n]. "
            "If the evidence is insufficient, answer exactly INSUFFICIENT_EVIDENCE.\n\n"
            f"Question: {query}\n\nEvidence:\n{context}"
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = str(response.json()["choices"][0]["message"]["content"]).strip()
        citation_ids = sorted({int(value) for value in CITATION_PATTERN.findall(text)})
        valid_ids = [value for value in citation_ids if 1 <= value <= len(evidence)]
        if len(valid_ids) != len(citation_ids):
            return GeneratedAnswer(text="", citation_ids=[])
        return GeneratedAnswer(text=text, citation_ids=valid_ids)


def materialize_citations(
    evidence: list[RetrievedChunk], citation_ids: list[int], quote_words: int = 42
) -> list[Citation]:
    citations: list[Citation] = []
    for citation_id in dict.fromkeys(citation_ids):
        item = evidence[citation_id - 1]
        quote = " ".join(item.chunk.text.split()[:quote_words])
        citations.append(
            Citation(
                citation_id=citation_id,
                chunk_id=item.chunk.chunk_id,
                source=item.chunk.source,
                title=item.chunk.title,
                section=item.chunk.section,
                quote=quote,
                retrieval_score=round(item.score, 6),
            )
        )
    return citations
