import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.clients.confluence_client import ConfluenceClient
from app.config.settings import Settings
from app.services.vector_store import VectorStoreRepository

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, settings: Settings, confluence_client: ConfluenceClient, vector_repo: VectorStoreRepository) -> None:
        self._settings = settings
        self._confluence_client = confluence_client
        self._vector_repo = vector_repo

    def ingest(self, force_reembed: bool = False) -> dict[str, bool | int]:
        if self._vector_repo.has_persisted_data() and not force_reembed:
            logger.info("Persisted index found. Skipping document fetch/chunking. Set force_reembed=true to rebuild.")
            self._vector_repo.ensure_ready()
            return {"documents_loaded": 0, "chunks_indexed": 0, "reused_existing": True}

        logger.info("Starting Confluence document loading. force_reembed=%s", force_reembed)
        raw_docs = self._confluence_client.load_documents()
        logger.info("Loaded %s Confluence documents.", len(raw_docs))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        split_docs: list[Document] = splitter.split_documents(raw_docs)
        logger.info("Chunking completed. chunks=%s", len(split_docs))

        index_result = self._vector_repo.index_documents(split_docs, force_reembed=force_reembed)
        logger.info("Vector indexing completed. reused_existing=%s", index_result["reused_existing"])

        return {
            "documents_loaded": len(raw_docs),
            "chunks_indexed": int(index_result["chunks_indexed"]),
            "reused_existing": bool(index_result["reused_existing"]),
        }
