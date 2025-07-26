"""
Retrieval layer combining BM25 and vector search.

This module encapsulates access to the Postgres/pgvector database and
implements hybrid retrieval by combining keyword search (full‑text
search using BM25) with semantic similarity search over dense
embeddings.  It also provides utilities to compute query embeddings
using either OpenAI or a local sentence‑transformers model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import asyncpg
import numpy as np  # type: ignore

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from .settings import Settings

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore


@dataclass
class DocumentChunk:
    """A chunk of a document retrieved from the database."""

    doc_id: str
    chunk_id: int
    content: str
    score: float


class Retriever:
    """Hybrid BM25 + vector retriever against Postgres/pgvector."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine: AsyncEngine = create_async_engine(settings.database_url, echo=False)
        self._model: Optional[SentenceTransformer] = None
        # Preload local embedding model if OpenAI key is not provided
        if not settings.openai_api_key and SentenceTransformer is not None:
            # Use a light-weight all-MiniLM model for local embeddings
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def _get_embedding(self, text: str) -> List[float]:
        """Compute an embedding for a text string using the configured backend."""
        if self.settings.openai_api_key and openai is not None:
            # Use OpenAI Embeddings API
            try:
                openai.api_key = self.settings.openai_api_key
                openai.base_url = self.settings.openai_api_base
                resp = await openai.Embedding.acreate(
                    input=[text],
                    model="text-embedding-ada-002",
                )  # type: ignore[attr-defined]
                return resp["data"][0]["embedding"]
            except Exception:
                # Fall back to local model if OpenAI fails
                pass
        if self._model is None:
            raise RuntimeError("No embedding model available; install sentence_transformers or set OPENAI_API_KEY")
        emb: List[float] = self._model.encode(text).tolist()  # type: ignore[no-untyped-call]
        return emb

    async def bm25_search(self, query: str, top_k: int) -> List[DocumentChunk]:
        """Perform a BM25 keyword search against the full‑text index."""
        sql = text(
            """
            SELECT d.doc_id, c.chunk_id, c.content,
                   ts_rank(c.tsv, plainto_tsquery('english', :query)) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.tsv @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :top_k
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"query": query, "top_k": top_k})
            rows = result.fetchall()
        return [DocumentChunk(r.doc_id, r.chunk_id, r.content, float(r.score)) for r in rows]

    async def vector_search(self, query_embedding: List[float], top_k: int) -> List[DocumentChunk]:
        """Perform a vector similarity search against the embedding index."""
        # Convert to Python list of float for query parameter
        sql = text(
            """
            SELECT d.doc_id, c.chunk_id, c.content,
                   (1 - (c.embedding <=> :embedding)) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.embedding <=> :embedding
            LIMIT :top_k
            """
        )
        async with self._engine.connect() as conn:
            # asyncpg will automatically cast Python list to pgvector
            result = await conn.execute(sql, {"embedding": query_embedding, "top_k": top_k})
            rows = result.fetchall()
        return [DocumentChunk(r.doc_id, r.chunk_id, r.content, float(r.score)) for r in rows]

    async def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        weight_bm25: Optional[float] = None,
        weight_vector: Optional[float] = None,
    ) -> List[DocumentChunk]:
        """Combine BM25 and vector search with configurable weights."""
        k = top_k or self.settings.retrieval_top_k
        w_bm25 = weight_bm25 if weight_bm25 is not None else self.settings.hybrid_weight_bm25
        w_vec = weight_vector if weight_vector is not None else self.settings.hybrid_weight_vector
        # Perform both searches concurrently
        emb = await self._get_embedding(query)
        bm25_task = asyncio.create_task(self.bm25_search(query, k))
        vector_task = asyncio.create_task(self.vector_search(emb, k))
        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)
        # Merge by computing combined score
        combined: Dict[Tuple[str, int], DocumentChunk] = {}
        for chunk in bm25_results:
            combined[(chunk.doc_id, chunk.chunk_id)] = DocumentChunk(
                chunk.doc_id,
                chunk.chunk_id,
                chunk.content,
                score=chunk.score * w_bm25,
            )
        for chunk in vector_results:
            key = (chunk.doc_id, chunk.chunk_id)
            if key in combined:
                combined[key].score += chunk.score * w_vec
            else:
                combined[key] = DocumentChunk(chunk.doc_id, chunk.chunk_id, chunk.content, chunk.score * w_vec)
        # Sort by combined score desc and return top_k
        sorted_chunks = sorted(combined.values(), key=lambda c: c.score, reverse=True)
        return sorted_chunks[:k]
