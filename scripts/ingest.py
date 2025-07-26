#!/usr/bin/env python3
"""
Ingest documents from the corpus into the Postgres/pgvector database.

This script iterates through all files in `data/corpus/`, extracts
text from PDFs and HTML files, chunks the text, computes embeddings
for each chunk and writes the result into the `documents` and
`chunks` tables.  The embedding dimension is determined by the
backend (OpenAI or sentence-transformers).

Usage:
    python scripts/ingest.py

The script reads configuration from the environment via
``app.settings.Settings``.  Ensure that the database service is
running (e.g. via `docker-compose up postgres`) before ingesting.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Tuple

import asyncpg
from pdfminer.high_level import extract_text  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

import sys
from pathlib import Path

# Ensure imports work when running this script directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "rag-industrial-eval-guardrails"))

from app.settings import Settings
from app.retriever import Retriever


def chunk_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks of approximately `chunk_size` tokens."""
    words = text.split()
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n:
            break
        start = end - overlap
    return chunks


def extract_text_from_file(path: Path) -> str:
    """Extract raw text from a PDF or HTML file."""
    if path.suffix.lower() == ".pdf":
        return extract_text(str(path))
    elif path.suffix.lower() in {".html", ".htm"}:
        with path.open("r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator=" ")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


async def ingest_corpus(settings: Settings) -> None:
    corpus_dir = Path(__file__).resolve().parents[1] / "data" / "corpus"
    if not corpus_dir.exists():
        print(f"Corpus directory {corpus_dir} does not exist. Run generate_synthetic_corpus.py first.")
        return
    # Connect to the database using asyncpg
    conn = await asyncpg.connect(
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
        host=settings.postgres_host,
        port=settings.postgres_port,
    )
    retriever = Retriever(settings)
    for file_path in sorted(corpus_dir.iterdir()):
        if not file_path.is_file():
            continue
        try:
            text = extract_text_from_file(file_path)
        except Exception as exc:
            print(f"Erreur lors de l'extraction du fichier {file_path}: {exc}")
            continue
        doc_id = file_path.name
        metadata = {"filename": file_path.name}
        # Insert or get document id
        doc_row = await conn.fetchrow(
            "INSERT INTO documents (doc_id, doc_name, content, metadata) VALUES ($1, $2, $3, $4) ON CONFLICT (doc_id) DO UPDATE SET content = EXCLUDED.content RETURNING id",
            doc_id,
            file_path.stem,
            text,
            json.dumps(metadata),
        )
        document_id = doc_row["id"]
        # Chunk text
        chunks = chunk_tokens(text, settings.chunk_size, settings.chunk_overlap)
        for idx, chunk_text in enumerate(chunks):
            # Compute embedding
            emb = await retriever._get_embedding(chunk_text)
            # Insert chunk
            await conn.execute(
                "INSERT INTO chunks (document_id, chunk_id, content, embedding, tsv) VALUES ($1, $2, $3, $4, to_tsvector('english', $3)) ON CONFLICT (document_id, chunk_id) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, tsv = EXCLUDED.tsv",
                document_id,
                idx,
                chunk_text,
                emb,
            )
        print(f"Ingested {file_path.name} with {len(chunks)} chunks.")
    await conn.close()


def main() -> None:
    settings = Settings()
    asyncio.run(ingest_corpus(settings))


if __name__ == "__main__":
    main()