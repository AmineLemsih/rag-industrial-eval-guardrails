import pytest
import sys
from pathlib import Path

# Append project path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "rag-industrial-eval-guardrails"))

from app.pii import PiiDetector
from app.guardrails import classify_question, validate_citations
from app.schemas import Citation


def test_pii_detection_masking():
    detector = PiiDetector()
    text = "Contactez moi à john.doe@example.com ou au 06 12 34 56 78."
    spans = detector.detect(text)
    assert any(span.entity_type == "EMAIL_ADDRESS" for span in spans)
    assert any(span.entity_type == "PHONE_NUMBER" for span in spans)
    masked = detector.mask(text, spans)
    # ensure sensitive info is masked
    assert "john" not in masked
    assert "06" not in masked


def test_classify_question_disallowed():
    assert classify_question("Quelle est votre opinion politique ?") is False
    assert classify_question("Quels sont les horaires de travail ?") is True


def test_validate_citations():
    citations = [Citation(doc_id="doc1", chunk_id=0), Citation(doc_id="doc2", chunk_id=1)]
    retrieved = {("doc1", 0): "texte", ("doc2", 1): "autre texte"}
    assert validate_citations(citations, retrieved) is True
    bad_citations = [Citation(doc_id="doc3", chunk_id=2)]
    assert validate_citations(bad_citations, retrieved) is False