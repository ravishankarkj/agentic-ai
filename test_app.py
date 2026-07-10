from app.api.dependencies import get_container, ServiceContainer
from app.logging_config import configure_logging
from app.api.schemas import QueryRequest, QueryResponse, SourceDocument

configure_logging()

container = get_container()

container_is_ready = container.vector_repo.is_ready()

print(f"Vector Repository Ready: {container_is_ready}")

def run_qa_service(request: QueryRequest, container: ServiceContainer) -> QueryResponse:
    answer, docs = container.qa_service.answer(request.query)

    sources = [
        SourceDocument(
            source=str(doc.metadata.get("source") or doc.metadata.get("id") or "unknown"),
            score=doc.metadata.get("score"),
            snippet=doc.page_content[:300],
        )
        for doc in docs
    ]

    response = QueryResponse(query=request.query, answer=answer, sources=sources)
    return response

response = run_qa_service(QueryRequest(query="What are the content types available in confluence?"), container)
print(f"QA Service Response: \n{response}")