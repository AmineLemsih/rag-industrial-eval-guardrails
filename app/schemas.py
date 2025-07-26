"""
API and internal data models.

This module defines Pydantic schemas for requests and responses
used by the FastAPI endpoints as well as internal representations
of documents and chunks.  Keeping these definitions in one place
helps maintain consistency between the various parts of the system.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for a query submitted by a client."""

    query: str = Field(..., description="Question posée par l'utilisateur")
    top_k: Optional[int] = Field(None, description="Nombre de passages à récupérer avant reranking")


class Citation(BaseModel):
    """Metadata describing a citation returned with an answer."""

    doc_id: str = Field(..., description="Identifiant unique du document")
    chunk_id: int = Field(..., description="Identifiant du passage dans le document")
    start: Optional[int] = Field(None, description="Position du début du passage utilisé")
    end: Optional[int] = Field(None, description="Position de fin du passage utilisé")


class QueryResponse(BaseModel):
    """Schema returned for a processed query."""

    answer: str = Field(..., description="Réponse générée par le modèle")
    citations: List[Citation] = Field(..., description="Liste des citations avec identifiants et offsets")
    cost: Optional[float] = Field(None, description="Coût estimé de la requête en dollars")
    latency_ms: Optional[float] = Field(None, description="Durée totale de traitement de la requête en millisecondes")


class IngestRequest(BaseModel):
    """Schema for ingestion via the API (optional)."""

    uri: str = Field(..., description="URI vers un document à ingérer")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Métadonnées associées au document")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Statut de santé du service")
