"""Small in-memory vector retriever for synthetic support policy documents."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Loads the embedding model only on the first encode call."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model: Any = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class RetrievedPolicy:
    id: str
    title: str
    category: str
    source: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "source": self.source,
            "text": self.text,
            "score": round(float(self.score), 4),
        }


class PolicyRetriever:
    def __init__(self, documents: list[dict[str, Any]], encoder: Encoder | None = None):
        if not documents:
            raise ValueError("At least one policy document is required.")
        self.documents = documents
        self.encoder = encoder or SentenceTransformerEncoder()
        self._document_vectors: np.ndarray | None = None

    def _vectors(self) -> np.ndarray:
        if self._document_vectors is None:
            texts = [f"{d['title']}. {d['category']}. {d['text']}" for d in self.documents]
            vectors = self.encoder.encode(texts)
            if vectors.ndim != 2 or vectors.shape[0] != len(self.documents):
                raise ValueError("Encoder returned an invalid document vector matrix.")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self._document_vectors = vectors / np.maximum(norms, 1e-12)
        return self._document_vectors

    def search(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty.")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10.")
        query_vector = np.asarray(self.encoder.encode([clean_query])[0], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 1e-12:
            raise ValueError("Encoder returned an empty query vector.")
        query_vector = query_vector / query_norm
        scores = self._vectors() @ query_vector
        ordered_indices = np.argsort(scores)[::-1][: min(top_k, len(self.documents))]
        results = []
        for idx in ordered_indices:
            doc = self.documents[int(idx)]
            results.append(
                RetrievedPolicy(
                    id=doc["id"],
                    title=doc["title"],
                    category=doc["category"],
                    source=doc["source"],
                    text=doc["text"],
                    score=float(scores[int(idx)]),
                )
            )
        return results
