"""
Re‑ranking of retrieved passages using a cross‑encoder.

This module wraps a sentence‑transformers cross‑encoder (such as
`bge-reranker-base` or `bge-reranker-large`) to score candidate
passages given the user's query.  The cross‑encoder operates on
query/document pairs and produces a single relevance score.  A
lighter identity reranker is provided when the heavy model is
unavailable.
"""

from __future__ import annotations

from typing import List

try:
    from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
except ImportError:
    CrossEncoder = None  # type: ignore

from .retriever import DocumentChunk
from .settings import Settings


class Reranker:
    """Wrapper around a cross‑encoder to rerank document chunks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.reranker_model
        self.model: CrossEncoder | None = None
        if CrossEncoder is not None:
            try:
                # Load the cross encoder with default settings.  By default
                # this will download weights on first use; consider caching
                # them in a volume for Docker builds.
                self.model = CrossEncoder(self.model_name)
            except Exception:
                # If loading fails, we keep None and fall back to identity ranking
                self.model = None

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int | None = None) -> List[DocumentChunk]:
        """Rerank candidate chunks according to the query.

        Parameters
        ----------
        query:
            User question.
        chunks:
            Candidate passages returned by the retriever.
        top_k:
            Optional number of passages to return after reranking.  If None,
            all passages are returned in new order.
        """
        if not chunks:
            return []
        if self.model is None:
            # Identity ranking if the cross‑encoder is not available
            return chunks[: (top_k or len(chunks))]
        # Prepare query-document pairs
        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(pairs)  # type: ignore[attr-defined]
        # Combine scores with the DocumentChunk objects
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[: (top_k or len(sorted_chunks))]
