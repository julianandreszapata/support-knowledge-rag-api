# Evaluación inicial

## Consultas de prueba

| Pregunta | Política esperada en top 3 |
|---|---|
| ¿Qué hago si el pedido llegó 20 minutos tarde? | POL-001 |
| Necesito cancelar una orden que ya salió del restaurante. | POL-002 |
| Me faltó un producto de la bolsa. | POL-003 |
| El envase llegó dañado y puede ser peligroso. | POL-004 |
| Quiero hablar con una persona. | POL-005 |
| ¿Puedo compartir la clave API para revisar el pedido? | POL-006 |

## Métricas a incorporar

- Recall@k: porcentaje de preguntas donde la política esperada aparece en los primeros k resultados.
- MRR: posición recíproca de la primera política relevante.
- Latencia de recuperación (ms).
- Porcentaje de respuestas con citas válidas.
- Tasa de fallback cuando no hay evidencia o el LLM no está disponible.

No reportar métricas hasta ejecutar una evaluación sobre un conjunto definido y conservar sus resultados. Los tests actuales validan el contrato y el ranking con un encoder determinista de prueba, no la calidad real del modelo de embeddings.
