"""
PII detection and anonymisation utilities.

The guardrails layer uses this module to identify personally
identifiable information (PII) within user queries or generated
answers.  It provides a simple rule‑based detector backed by
regular expressions as well as an optional integration with the
`presidio_analyzer` library for more comprehensive detection when
available.  If Presidio is not installed or fails to initialise
(for example because its spaCy model is missing), the code falls
back to a handful of regex patterns for common PII such as
email addresses and telephone numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer import RecognizerResult
    _HAS_PRESIDIO = True
except ImportError:
    AnalyzerEngine = None  # type: ignore
    RecognizerResult = None  # type: ignore
    _HAS_PRESIDIO = False


@dataclass
class PiiSpan:
    """A detected PII span with its type and confidence."""

    start: int
    end: int
    entity_type: str
    score: float


class PiiDetector:
    """Detect PII using either Presidio or simple regexes."""

    # Basic patterns to fall back on if Presidio is unavailable
    EMAIL_PATTERN = re.compile(r"[\w.-]+@[\w.-]+\.[A-Za-z]{2,}")
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    def __init__(self) -> None:
        self.presidio: Optional[AnalyzerEngine] = None
        if _HAS_PRESIDIO:
            try:
                # Initialise Presidio with default recognisers.
                self.presidio = AnalyzerEngine()
            except Exception:
                # Presidio failed to initialise (e.g. missing spaCy model)
                self.presidio = None

    def detect(self, text: str) -> List[PiiSpan]:
        """Return a list of PiiSpan detected in the given text."""
        spans: List[PiiSpan] = []
        if self.presidio is not None:
            results: List[RecognizerResult] = self.presidio.analyze(text=text, language="en")  # type: ignore[call-arg]
            for res in results:
                spans.append(PiiSpan(start=res.start, end=res.end, entity_type=res.entity_type, score=res.score))
        else:
            # Simple regex fallback
            for match in self.EMAIL_PATTERN.finditer(text):
                spans.append(PiiSpan(match.start(), match.end(), "EMAIL_ADDRESS", 0.8))
            for match in self.PHONE_PATTERN.finditer(text):
                spans.append(PiiSpan(match.start(), match.end(), "PHONE_NUMBER", 0.7))
            for match in self.CREDIT_CARD_PATTERN.finditer(text):
                spans.append(PiiSpan(match.start(), match.end(), "CREDIT_CARD", 0.9))
        return spans

    def mask(self, text: str, spans: List[PiiSpan]) -> str:
        """Return the text with detected PII masked by asterisks."""
        masked = list(text)
        for span in spans:
            for i in range(span.start, span.end):
                if 0 <= i < len(masked):
                    masked[i] = "*"
        return "".join(masked)
