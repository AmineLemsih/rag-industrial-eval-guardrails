import pytest
import asyncpg
import asyncio
import sys
from pathlib import Path

# Append project path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "rag-industrial-eval-guardrails"))

from app.settings import Settings
from app.retriever import Retriever
from scripts.generate_synthetic_corpus import main as gen_corpus
from scripts.ingest import ingest_corpus


@pytest.mark.asyncio
async def test_hybrid_retriever_returns_results():
    settings = Settings()
    try:
        conn = await asyncpg.connect(
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            host=settings.postgres_host,
            port=settings.postgres_port,
        )
        await conn.close()
    except Exception:
        pytest.skip("Postgres is not available")
    # Generate and ingest
    gen_corpus()
    await ingest_corpus(settings)
    retriever = Retriever(settings)
    results = await retriever.hybrid_search("mot de passe", top_k=3)
    assert len(results) > 0
    # Ensure scores are sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)