"""
Application package for the RAG industrial pipeline.

This package wires together the components of the retrieval‑augmented
generation system including settings management, database access,
retrieval, reranking, guardrails and the API layer.  Modules are
factored to promote testability and decoupling between the stages of
the pipeline.
"""

__all__ = [
    "settings",
    "schemas",
    "retriever",
    "reranker",
    "guardrails",
    "pii",
    "rag_pipeline",
]