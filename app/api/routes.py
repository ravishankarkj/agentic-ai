from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, QueryResult, SourceDocument

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, container: ServiceContainer = Depends(get_container)) -> IngestResponse:
    try:
        result = container.ingestion_service.ingest(force_reembed=request.force_reembed)
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    if not container.vector_repo.is_ready():
        raise HTTPException(status_code=400, detail="No indexed documents found. Call /api/v1/ingest first.")

    answer, docs = container.qa_service.answer(request.query)

    sources = [
        SourceDocument(
            source=str(doc.metadata.get("source") or doc.metadata.get("id") or "unknown"),
            score=doc.metadata.get("score"),
            snippet=doc.page_content[:300],
        )
        for doc in docs
    ]

    response = QueryResponse(
        requestId=request.requestId,
        userId=request.userId,
        response=QueryResult(query=request.query, answer=answer, sources=sources),
    )

    # Placeholder outbound transport for downstream integration.
    await container.response_dispatcher.dispatch(response.model_dump())

    return response
