# Brain Box With AI

A local-first RAG application with:

- FastAPI backend for REST APIs
- Confluence document ingestion
- In-memory vector database
- Hybrid retrieval (dense + sparse BM25)
- LangChain-based QA pipeline
- Gradio test UI
- Interchangeable LLM provider abstraction (default: Ollama Cloud + Gemma)

## Architecture

- app/config: environment-driven settings
- app/clients: external system clients (Confluence, LLM factory)
- app/services: ingestion, vector store, retrieval, QA orchestration, response dispatch
- app/api: FastAPI app, routes, and dependencies
- app/ui: Gradio application (separate from API)

## Prerequisites

- Python 3.12+
- Access to Confluence
- Access to Ollama-compatible endpoint and model

## Setup

1. Install dependencies.
2. Copy .env.example to .env and set required values.
3. Run the FastAPI app locally.
4. Optionally run Gradio for interactive testing.

## Run

- API: uv run run-api
- API (alternative): uv run python -m app.main
- Gradio: uv run run-gradio

By default API runs on localhost:8000 and Gradio on localhost:7860.

## Deploy To Render

This repository includes [render.yaml](render.yaml) for Render Blueprint deployment.

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and select the repository.
3. Render will apply `render.yaml` and create the web service.
4. Add all required environment variables in Render dashboard (Confluence, LLM, callback, etc.).

Key runtime behavior:

- API host binds to `HOST` (default `0.0.0.0`).
- API port binds to `PORT` (required by Render).
- Start command is `uv run python -m app.main`.

## REST Endpoints

- GET /health
- POST /api/v1/ingest (load Confluence docs into memory)
- POST /api/v1/query (retrieve context, generate answer, trigger optional outbound HTTP callback)

Ingest request body:

```json
{
	"force_reembed": false
}
```

## Notes

- The vector store uses Chroma persistence under `VECTOR_STORE_DIR` and is reused across restarts.
- Set `force_reembed=true` in ingest requests only when you need to rebuild embeddings.
- Outbound response dispatch is implemented with a placeholder callback URL in environment settings.
- LLM provider is interchangeable through LLM_PROVIDER in .env.
