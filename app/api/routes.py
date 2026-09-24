from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, QueryResult, SourceDocument

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_sources(docs) -> list[SourceDocument]:
    sources: list[SourceDocument] = []
    for doc in docs:
        link = doc.metadata.get("source")
        label = str(doc.metadata.get("title") or doc.metadata.get("id") or link or "unknown")
        sources.append(
            SourceDocument(
                source=label,
                link=str(link) if link else None,
                score=doc.metadata.get("score"),
                snippet=doc.page_content[:300],
            )
        )
    return sources


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

    session_id = request.sessionId or request.userId
    history = container.conversation_store.get_history(session_id)
    standalone_query = await container.qa_service.condense_question(request.query, history)
    logger.info("Standalone query generated: %s", standalone_query)

    docs = container.qa_service.retrieve_context(standalone_query)
    sources = _build_sources(docs)

    answer_chunks: list[str] = []
    sequence = 0
    async for chunk in container.qa_service.stream_answer(standalone_query, docs, history):
        answer_chunks.append(chunk)
        # logger.info("Streaming chunk received: %s", chunk)
        # await container.response_dispatcher.dispatch_stream_chunk(
        #     request_id=request.requestId,
        #     user_id=request.userId,
        #     query=request.query,
        #     chunk=chunk,
        #     sequence=sequence,
        # )
        # sequence += 1

    answer = "".join(answer_chunks)
    # logger.info("answer = \n\n%s", answer)
    # logger.info("Query = \n\n%s", request.query)

    container.conversation_store.append_turn(session_id, request.query, answer)

    response = QueryResponse(
        requestId=request.requestId,
        userId=request.userId,
        sessionId=session_id,
        response=QueryResult(
            query=request.query,
            standaloneQuery=standalone_query,
            answer=answer,
            sources=sources,
        ),
    )

    await container.response_dispatcher.dispatch_sources(
        request_id=request.requestId,
        user_id=request.userId,
        query=request.query,
        answer=answer,
        sources=[source.model_dump() for source in sources],
    )

    return response


@router.delete("/api/v1/session/{session_id}")
def reset_session(session_id: str, container: ServiceContainer = Depends(get_container)) -> dict[str, str]:
    container.conversation_store.clear(session_id)
    return {"status": "cleared", "sessionId": session_id}
