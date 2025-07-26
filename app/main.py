"""
FastAPI application entry point.

Exposes health, query, ingestion and metrics endpoints.  The query
endpoint enforces rate limiting and delegates processing to the RAG
pipeline.  Metrics are collected via `prometheus_client` and include
latency histograms and request counters.
"""

from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .settings import Settings
from .schemas import QueryRequest, QueryResponse, IngestRequest, HealthResponse
from .rag_pipeline import answer_query
from .guardrails import RateLimiter


settings = Settings()
app = FastAPI(title=settings.app_name)
rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_period)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "rag_api_requests_total", "Total number of requests", ["endpoint"]
)
REQUEST_LATENCY = Histogram(
    "rag_api_request_latency_seconds", "Latency of requests in seconds", ["endpoint"]
)


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, payload: QueryRequest) -> QueryResponse:
    """Process a question through the RAG pipeline and return an answer."""
    client_id = request.client.host if request.client else "anonymous"
    allowed = await rate_limiter.allow(client_id)
    if not allowed:
        raise HTTPException(status_code=429, detail="Trop de requêtes, veuillez réessayer plus tard.")
    REQUEST_COUNT.labels(endpoint="query").inc()
    with REQUEST_LATENCY.labels(endpoint="query").time():
        response = await answer_query(payload.query, settings)
    return response


@app.post("/ingest")
async def ingest_endpoint(payload: IngestRequest) -> JSONResponse:
    """Ingest a remote document.  This endpoint is a stub that delegates to the ingestion script.

    In a real deployment this should download the document from the URI,
    extract its text and insert it into the database.  For the purposes
    of this project we instruct users to run the ingestion script via
    `make ingest` instead.
    """
    return JSONResponse(
        status_code=501,
        content={"detail": "L'ingestion via l'API n'est pas implémentée. Utilisez make ingest."},
    )


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scraping."""
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
