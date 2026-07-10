import logging
import json
from pathlib import Path
from threading import Lock

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStoreRepository:
    def __init__(
        self,
        embeddings: Embeddings,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        persist_directory: str,
        collection_name: str,
    ) -> None:
        logging.info("Top K: %s", top_k)
        self._embeddings = embeddings
        self._top_k = top_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._sparse_cache_path = Path(self._persist_directory) / f"{self._collection_name}_sparse_cache.json"

        self._lock = Lock()
        self._vector_store: Chroma | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._hybrid_retriever: EnsembleRetriever | None = None

    def initialize(self) -> None:
        with self._lock:
            self._ensure_vector_store()
            self._hydrate_sparse_retriever_from_store()
            self._build_hybrid_retriever_if_ready()

    def _ensure_vector_store(self) -> None:
        if self._vector_store is not None:
            return

        Path(self._persist_directory).mkdir(parents=True, exist_ok=True)
        self._vector_store = Chroma(
            collection_name=self._collection_name,
            persist_directory=self._persist_directory,
            embedding_function=self._embeddings,
        )
        logger.info(
            "Initialized Chroma vector store. directory=%s collection=%s",
            self._persist_directory,
            self._collection_name,
        )

    def _save_sparse_cache(self, docs: list[Document]) -> None:
        payload: list[dict] = []
        for doc in docs:
            payload.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "id": doc.id,
                }
            )

        self._sparse_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._sparse_cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        logger.info("Updated sparse cache at %s with %s chunks.", self._sparse_cache_path, len(payload))

    def _load_sparse_cache(self) -> list[Document]:
        if not self._sparse_cache_path.exists():
            return []

        try:
            raw = json.loads(self._sparse_cache_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse sparse cache at %s. Rebuilding from Chroma.", self._sparse_cache_path)
            return []

        docs: list[Document] = []
        for item in raw:
            if not item.get("page_content"):
                continue
            docs.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata") or {},
                    id=item.get("id"),
                )
            )

        if docs:
            logger.info("Loaded %s chunks from sparse cache.", len(docs))
        return docs

    def _get_all_documents_from_store(self) -> list[Document]:
        self._ensure_vector_store()
        assert self._vector_store is not None

        payload = self._vector_store.get(include=["documents", "metadatas"])
        docs = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []

        hydrated: list[Document] = []
        for index, page_content in enumerate(docs):
            if not page_content:
                continue
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            hydrated.append(Document(page_content=page_content, metadata=metadata))

        return hydrated

    def _hydrate_sparse_retriever_from_store(self) -> None:
        docs = self._load_sparse_cache()
        if not docs:
            docs = self._get_all_documents_from_store()
            if docs:
                self._save_sparse_cache(docs)
        if not docs:
            self._bm25_retriever = None
            logger.info("No persisted vectors found in Chroma yet.")
            return

        self._bm25_retriever = BM25Retriever.from_documents(docs)
        self._bm25_retriever.k = self._top_k
        logger.info("Hydrated BM25 retriever from persisted store with %s chunks.", len(docs))

    def _build_hybrid_retriever_if_ready(self) -> None:
        if self._vector_store is None or self._bm25_retriever is None:
            self._hybrid_retriever = None
            return

        dense_retriever = self._vector_store.as_retriever(search_kwargs={"k": self._top_k})
        self._hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25_retriever],
            weights=[self._dense_weight, self._sparse_weight],
        )
        logger.info(
            "Initialized hybrid retriever with dense_weight=%s sparse_weight=%s top_k=%s.",
            self._dense_weight,
            self._sparse_weight,
            self._top_k,
        )

    def has_persisted_data(self) -> bool:
        # with self._lock:
            # docs = self._get_all_documents_from_store()
            # return len(docs) > 0
        try:
            return self._vector_store._collection.count() > 0  # noqa: SLF001 - Chroma API
        except Exception:  # pragma: no cover - defensive
            return False

    def _count(self) -> int:
        """Return the number of embedded chunks currently persisted."""

        self._ensure_vector_store()
        assert self._vector_store is not None

        try:
            return self._vector_store._collection.count()  # noqa: SLF001 - Chroma API
        except Exception:  # pragma: no cover - defensive
            return 0

    def _clear_store(self) -> None:
        """Delete all embedded content in the collection."""

        self._ensure_vector_store()
        assert self._vector_store is not None

        payload = self._vector_store.get(include=[])
        ids = payload.get("ids") or []
        if ids:
            self._vector_store.delete(ids=ids)
            logger.info("Cleared %s existing chunks from Chroma before re-embedding.", len(ids))

        if self._sparse_cache_path.exists():
            self._sparse_cache_path.unlink()
            logger.info("Deleted sparse cache file at %s", self._sparse_cache_path)

    def index_documents(self, docs: list[Document], force_reembed: bool = False) -> dict[str, bool | int]:
        if not docs:
            raise ValueError("No documents provided for indexing")

        with self._lock:
            self._ensure_vector_store()
            assert self._vector_store is not None

            has_existing = self.has_persisted_data()
            if has_existing and not force_reembed:
                logger.info("Using persisted Chroma index. Skipping re-embedding.")
                self._hydrate_sparse_retriever_from_store()
                self._build_hybrid_retriever_if_ready()
                existing_chunks = self._count()
                return {
                    "documents_loaded": 0,
                    "chunks_indexed": existing_chunks,
                    "reused_existing": True,
                }

            if has_existing and force_reembed:
                logger.info("Force re-embed requested. Rebuilding persisted Chroma index.")
                self._clear_store()

            logger.info("Indexing %s chunks into Chroma.", len(docs))
            self._vector_store.add_documents(docs)
            self._save_sparse_cache(docs)

            self._bm25_retriever = BM25Retriever.from_documents(docs)
            self._bm25_retriever.k = self._top_k
            self._build_hybrid_retriever_if_ready()

            return {
                "documents_loaded": 0,
                "chunks_indexed": len(docs),
                "reused_existing": False,
            }

    def ensure_ready(self) -> None:
        with self._lock:
            self._ensure_vector_store()
            if self._hybrid_retriever is None:
                self._hydrate_sparse_retriever_from_store()
                self._build_hybrid_retriever_if_ready()

    def is_ready(self) -> bool:
        return self._hybrid_retriever is not None

    def get_hybrid_retriever(self) -> EnsembleRetriever:
        self.ensure_ready()
        if not self.is_ready():
            raise ValueError("Vector store is not initialized. Ingest documents first.")

        assert self._hybrid_retriever is not None
        return self._hybrid_retriever
