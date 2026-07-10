import logging

from langchain_core.documents import Document

from app.services.vector_store import VectorStoreRepository

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, vector_repo: VectorStoreRepository) -> None:
        self._vector_repo = vector_repo

    def retrieve(self, query: str) -> list[Document]:
        logger.info("Starting hybrid retrieval for query.")
        retriever = self._vector_repo.get_hybrid_retriever()
        docs = retriever.invoke(query)
        logger.info("Hybrid retrieval completed. documents=%s", len(docs))
        return docs
