import asyncio
import os
import sys
import pytest
from httpx import AsyncClient  # type: ignore
from pathlib import Path

# Ensure the project package can be imported when the repository name contains hyphens.
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "rag-industrial-eval-guardrails"))

from app.main import app, settings
from app.rag_pipeline import answer_query
from scripts.generate_synthetic_corpus import main as gen_corpus
from scripts.ingest import ingest_corpus


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_query_endpoint_end_to_end(tmp_path):
    """End to end test requiring a running Postgres.  Skips if the database is unreachable."""
    # Check connectivity to Postgres
    import asyncpg
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
    # Generate and ingest corpus
    gen_corpus()
    await ingest_corpus(settings)
    # Issue a query via the API
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/query", json={"query": "Quelle est la durée de validité des mots de passe ?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        # Should contain at least one citation
        assert isinstance(data.get("citations"), list)