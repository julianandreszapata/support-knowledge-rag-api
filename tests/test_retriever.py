from __future__ import annotations

import numpy as np
import pytest

from app.retriever import PolicyRetriever


class ToyEncoder:
    """Deterministic encoder for unit tests; no model download required."""
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            if any(term in lower for term in ("demora", "entrega tardía", "delay")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(term in lower for term in ("cancelación", "cancelado", "cancellation")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


POLICIES = [
    {"id": "POL-001", "title": "Demora en la entrega", "category": "delay", "source": "delay.md", "text": "Política de demora en la entrega tardía."},
    {"id": "POL-002", "title": "Pedido cancelado", "category": "cancel", "source": "cancel.md", "text": "Política sobre cancelación del pedido."},
    {"id": "POL-003", "title": "Privacidad", "category": "privacy", "source": "privacy.md", "text": "Protección de claves y datos personales."},
]


def test_search_returns_most_relevant_document_first() -> None:
    retriever = PolicyRetriever(POLICIES, encoder=ToyEncoder())
    results = retriever.search("demora en la entrega tardía", top_k=2)
    assert results[0].id == "POL-001"
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_search_rejects_empty_query() -> None:
    retriever = PolicyRetriever(POLICIES, encoder=ToyEncoder())
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("   ")


def test_search_validates_top_k() -> None:
    retriever = PolicyRetriever(POLICIES, encoder=ToyEncoder())
    with pytest.raises(ValueError, match="top_k must be between"):
        retriever.search("demora", top_k=0)
