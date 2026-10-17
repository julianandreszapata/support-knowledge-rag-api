# Arquitectura

```text
Cliente HTTP / Swagger UI
          |
          v
     FastAPI API
       |      \
       |       \ POST /answer (opcional)
       v        v
   Vector retriever ----> Gemini API (opcional)
       |
       v
Embeddings locales (sentence-transformers/all-MiniLM-L6-v2)
       |
       v
Políticas sintéticas versionadas en data/policies.json

Respuesta: evidencia recuperada + score + fuente + cita POL-xxx
```

## Flujo RAG

1. Cargar políticas sintéticas con IDs, títulos, categorías y fuentes.
2. Calcular embeddings de cada política con Sentence Transformers y normalizar vectores.
3. Convertir la pregunta del usuario en un embedding.
4. Calcular similitud coseno y recuperar los fragmentos más cercanos.
5. Para `/answer`, proporcionar esos fragmentos como contexto del LLM y pedir citas por ID.
6. Si la API del modelo no está disponible o no hay clave, retornar un modo `retrieval_only` con las fuentes recuperadas en lugar de bloquear la API.

## Limitaciones intencionales

- Dataset de demostración, no políticas de una empresa real.
- Los vectores se guardan en memoria; no es una base vectorial persistente ni un diseño para alto volumen.
- No implementa autenticación, multi-tenancy ni actualización automática de documentos.
- Los scores de similitud no equivalen a una probabilidad de corrección.
- Gemini es opcional; el endpoint de búsqueda se puede demostrar sin consumir cuota del LLM.

## Evolución sugerida

PostgreSQL + pgvector/Supabase, versionado de documentos, filtros por vigencia/país, autenticación, logs estructurados, evaluación offline y una suite de pruebas de recuperación.
