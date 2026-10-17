"""FastAPI endpoints for policy retrieval and optional grounded answer generation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.retriever import PolicyRetriever

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "data" / "policies.json").open("r", encoding="utf-8") as file:
    POLICIES: list[dict[str, Any]] = json.load(file)

retriever = PolicyRetriever(POLICIES)
app = FastAPI(
    title="Support Knowledge RAG API",
    description=(
        "Demostración de recuperación vectorial de políticas sintéticas de soporte. "
        "La generación con Gemini es opcional; /search funciona sin una clave de LLM."
    ),
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000, description="Pregunta de soporte")
    top_k: int = Field(default=3, ge=1, le=10)


class AnswerRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "support-knowledge-rag-api"}


@app.get("/policies")
def list_policies() -> dict[str, Any]:
    return {
        "count": len(POLICIES),
        "policies": [
            {"id": p["id"], "title": p["title"], "category": p["category"], "source": p["source"]}
            for p in POLICIES
        ],
    }


@app.post("/search")
def search_policies(body: SearchRequest) -> dict[str, Any]:
    try:
        results = retriever.search(body.query, body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Do not include credentials or environment values in error responses.
        raise HTTPException(
            status_code=503,
            detail="Policy embeddings are unavailable. Check model download/network and server logs.",
        ) from exc
    return {"query": body.query, "results": [item.to_dict() for item in results]}


def _grounded_fallback(query: str, results: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "query": query,
        "answer": (
            "No pude generar una respuesta con el modelo de lenguaje. Para evitar inventar una política, "
            "revisa los fragmentos recuperados y sus fuentes antes de responder al usuario."
        ),
        "generation_mode": "retrieval_only",
        "warning": reason,
        "citations": [
            {"id": r["id"], "title": r["title"], "source": r["source"], "score": r["score"]}
            for r in results
        ],
        "evidence": results,
    }


@app.post("/answer")
def answer_with_retrieval(body: AnswerRequest) -> dict[str, Any]:
    try:
        retrieved = retriever.search(body.query, body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Policy embeddings are unavailable. Check model download/network and server logs.",
        ) from exc
    results = [item.to_dict() for item in retrieved]
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _grounded_fallback(body.query, results, "GEMINI_API_KEY is not configured.")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=15000),
        )
        evidence = "\n\n".join(
            f"[{r['id']}] {r['title']} (fuente: {r['source']}): {r['text']}" for r in results
        )
        prompt = (
            "Eres un asistente de soporte. Responde en español usando únicamente la evidencia entregada. "
            "No inventes políticas, aprobaciones, reembolsos ni acciones realizadas. Si la evidencia es insuficiente, "
            "indícalo y recomienda revisión humana. Incluye IDs de política entre corchetes como citas.\n\n"
            f"Pregunta: {body.query}\n\nEvidencia recuperada:\n{evidence}"
        )
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=prompt,
        )
        output_text = (getattr(response, "text", None) or "").strip()
        if not output_text:
            return _grounded_fallback(body.query, results, "The model returned an empty answer.")
        return {
            "query": body.query,
            "answer": output_text,
            "generation_mode": "retrieval_augmented_generation",
            "citations": [
                {"id": r["id"], "title": r["title"], "source": r["source"], "score": r["score"]}
                for r in results
            ],
            "evidence": results,
        }
    except Exception as exc:
        # Graceful fallback keeps the API useful when quota/network is temporarily unavailable.
        return _grounded_fallback(
            body.query,
            results,
            f"LLM generation unavailable ({type(exc).__name__}); retrieved evidence is still available.",
        )
