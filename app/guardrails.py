"""
Guardrails for query processing and response generation.

This module collects functions that enforce safety and
compliance constraints on incoming questions and generated
responses.  Guardrails include:

* **PII detection and masking** – see :mod:`app.pii`.
* **Domain classification** to reject questions outside of the
  ingested corpus or that request disallowed content.
* **Citation validation** to ensure that the answer refers only to
  passages that were retrieved and returned to the model.
* **Rate limiting** to prevent abuse of the API.  The implementation
  here is a simple in‑memory token bucket, suitable for single
  instance deployments.  In production a distributed rate limiter
  should be used.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Dict, Tuple, Optional

from .pii import PiiDetector, PiiSpan
from .schemas import Citation


# Example list of topics considered out of scope.  In a real system this
# could be replaced by a classifier or curated list of allowed topics.
DISALLOWED_TOPICS = [
    "politics",
    "violence",
    "weapon",
    "hate speech",
    "adult",
]


class RateLimiter:
    """Simple token bucket rate limiter per client.

    This implementation stores counters in memory keyed by the client
    identifier.  It is suitable for a single‑process deployment.  In
    distributed setups (multiple API workers) a shared store such as
    Redis should be used instead.
    """

    def __init__(self, requests: int, period: float) -> None:
        self.capacity = requests
        self.period = period
        self.tokens: Dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def allow(self, client_id: str) -> bool:
        """Return True if a request from *client_id* is allowed."""
        async with self.lock:
            now = time.monotonic()
            tokens, last_time = self.tokens.get(client_id, (self.capacity, now))
            # Refill tokens
            elapsed = now - last_time
            refill = (elapsed / self.period) * self.capacity
            tokens = min(self.capacity, tokens + refill)
            if tokens >= 1:
                tokens -= 1
                self.tokens[client_id] = (tokens, now)
                return True
            else:
                self.tokens[client_id] = (tokens, now)
                return False


def classify_question(question: str) -> bool:
    """Return True if the question is allowed, False otherwise.

    A very naive classifier that flags questions containing any
    disallowed topic keyword.  This can be replaced by a more
    sophisticated semantic classifier or finite set of allowed domains.
    """
    lower_q = question.lower()
    for topic in DISALLOWED_TOPICS:
        if topic in lower_q:
            return False
    return True


def validate_citations(citations: List[Citation], retrieved_docs: Dict[Tuple[str, int], str]) -> bool:
    """Check that every citation references a known retrieved passage.

    Parameters
    ----------
    citations:
        List of citations returned by the model.
    retrieved_docs:
        Mapping from (doc_id, chunk_id) to the exact text of that chunk.

    Returns
    -------
    bool
        True if all citations refer to valid retrieved chunks, False otherwise.
    """
    for citation in citations:
        key = (citation.doc_id, citation.chunk_id)
        if key not in retrieved_docs:
            return False
    return True


def mask_pii_in_text(text: str, pii_detector: Optional[PiiDetector] = None) -> str:
    """Mask any detected PII in the provided text using the configured detector."""
    detector = pii_detector or PiiDetector()
    spans = detector.detect(text)
    return detector.mask(text, spans)
