"""
High level retrieval‑augmented generation pipeline.

This module coordinates the retrieval, reranking, guardrails, prompt
construction and call to the chosen language model.  It exposes a
single async function :func:`answer_query` which takes a user
question and returns an answer with citations and metadata.
"""

from __future__ import annotations

import asyncio
import time
import re
from typing import List, Dict, Tuple

from .settings import Settings
from .retriever import Retriever, DocumentChunk
from .reranker import Reranker
from .guardrails import PiiDetector, classify_question, mask_pii_in_text, validate_citations
from .schemas import Citation, QueryResponse

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    import aiohttp  # type: ignore
except ImportError:
    aiohttp = None  # type: ignore


async def generate_answer_via_openai(
    query: str, contexts: List[DocumentChunk], settings: Settings
) -> str:
    """Call OpenAI ChatCompletion API to generate an answer with citations."""
    messages = []
    # System prompt instructs the model to answer using only provided context
    system_prompt = (
        "Vous êtes un assistant qui répond exclusivement à partir des documents fournis. "
        "Citez toujours vos sources en indiquant les identifiants de passage (doc_id:chunk_id) "
        "utilisés dans la réponse. Si l'information demandée n'apparaît pas dans les contextes, "
        "répondez poliment que vous ne savez pas."
    )
    messages.append({"role": "system", "content": system_prompt})
    # Build context string
    context_lines = []
    for chunk in contexts:
        tag = f"[{chunk.doc_id}:{chunk.chunk_id}]"
        context_lines.append(f"{tag} {chunk.content.strip()}")
    context_text = "\n".join(context_lines)
    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Répondez à la question en utilisant uniquement le contexte ci-dessus et en citant les passages pertinents."
    )
    messages.append({"role": "user", "content": user_prompt})
    openai.api_key = settings.openai_api_key
    openai.base_url = settings.openai_api_base
    response = await openai.ChatCompletion.acreate(  # type: ignore[attr-defined]
        model=settings.default_model.split(":", 1)[1] if ":" in settings.default_model else settings.default_model,
        messages=messages,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"].strip()


async def generate_answer_via_local(
    query: str, contexts: List[DocumentChunk], settings: Settings
) -> str:
    """Fallback answer generation when remote LLMs are unavailable.

    This naive implementation simply returns the first retrieved
    chunk as the answer and cites it.  It ensures the API remains
    functional without external dependencies but provides no
    reasoning.  For production, plug in a local LLM or call
    Bedrock via boto3.
    """
    if not contexts:
        return "Je suis désolé, je ne trouve pas d'informations pertinentes dans le corpus."
    first = contexts[0]
    return f"{first.content.strip()}\n\n[Citation: {first.doc_id}:{first.chunk_id}]"


async def answer_query(query: str, settings: Settings) -> QueryResponse:
    """Process a user question through retrieval, reranking and generation.

    Parameters
    ----------
    query:
        The question asked by the user.
    settings:
        Application settings controlling models and thresholds.

    Returns
    -------
    QueryResponse
        The generated answer, list of citations and cost/latency metadata.
    """
    start_time = time.perf_counter()
    # Guardrail: classify the question
    if not classify_question(query):
        # Return a canned refusal if question is disallowed
        return QueryResponse(
            answer="Je suis désolé, je ne suis pas autorisé à répondre à cette question.",
            citations=[],
            cost=0.0,
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )
    # Detect PII in query and mask it before retrieval to avoid storing sensitive info
    pii_detector = PiiDetector()
    masked_query = mask_pii_in_text(query, pii_detector=pii_detector)
    # Run retrieval
    retriever = Retriever(settings)
    retrieved = await retriever.hybrid_search(masked_query)
    # Rerank
    reranker = Reranker(settings)
    reranked = reranker.rerank(masked_query, retrieved)
    # Select a subset of contexts (top 4) for generation
    contexts = reranked[:4]
    # Generate answer
    answer_text: str
    try:
        if settings.default_model.startswith("openai") and settings.openai_api_key and openai is not None:
            answer_text = await generate_answer_via_openai(masked_query, contexts, settings)
        else:
            # Local fallback
            answer_text = await generate_answer_via_local(masked_query, contexts, settings)
    except Exception:
        # In case of error with remote model, fallback locally
        answer_text = await generate_answer_via_local(masked_query, contexts, settings)
    # Detect PII in answer and mask
    answer_text = mask_pii_in_text(answer_text, pii_detector=pii_detector)
    # Parse citations from answer; look for patterns like [doc_id:chunk_id]
    citation_pattern = re.compile(r"\[(?P<doc>[\w/.-]+):(?P<chunk>\d+)\]")
    citations: List[Citation] = []
    for match in citation_pattern.finditer(answer_text):
        citations.append(Citation(doc_id=match.group("doc"), chunk_id=int(match.group("chunk"))))
    # Validate citations (optional) – build map of retrieved contexts
    retrieved_map: Dict[Tuple[str, int], str] = {(c.doc_id, c.chunk_id): c.content for c in contexts}
    if not validate_citations(citations, retrieved_map):
        # If citations are invalid, strip them and return refusal
        answer_text = "Je suis désolé, je ne peux pas fournir de réponse fiable car les citations sont invalides."
        citations = []
    latency = (time.perf_counter() - start_time) * 1000.0
    # Cost estimation – for demonstration we estimate embeddings and LLM usage
    # These numbers are placeholders and should be updated based on real API pricing.
    num_tokens = sum(len(chunk.content.split()) for chunk in contexts)
    cost = 0.0005 * num_tokens / 1000.0  # Rough cost per 1k tokens
    return QueryResponse(answer=answer_text, citations=citations, cost=cost, latency_ms=latency)
