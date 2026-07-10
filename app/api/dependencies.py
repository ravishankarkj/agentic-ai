from dataclasses import dataclass
from functools import lru_cache

from app.clients.confluence_client import ConfluenceClient
from app.clients.llm_factory import LLMFactory
from app.config.settings import Settings, get_settings
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService
from app.services.response_dispatcher import ResponseDispatcher
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStoreRepository


@dataclass(frozen=True)
class ServiceContainer:
    settings: Settings
    ingestion_service: IngestionService
    qa_service: QAService
    vector_repo: VectorStoreRepository
    response_dispatcher: ResponseDispatcher


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    settings = get_settings()

    llm_factory = LLMFactory(settings)
    embeddings = llm_factory.create_embeddings()
    chat_model = llm_factory.create_chat_model()

    confluence_client = ConfluenceClient(settings)
    vector_repo = VectorStoreRepository(
        embeddings=embeddings,
        top_k=settings.top_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        persist_directory=settings.vector_store_dir,
        collection_name=settings.vector_collection_name,
    )
    vector_repo.initialize()

    ingestion_service = IngestionService(
        settings=settings,
        confluence_client=confluence_client,
        vector_repo=vector_repo,
    )
    retrieval_service = RetrievalService(vector_repo=vector_repo)
    qa_service = QAService(retrieval_service=retrieval_service, chat_model=chat_model)
    dispatcher = ResponseDispatcher(settings=settings)

    return ServiceContainer(
        settings=settings,
        ingestion_service=ingestion_service,
        qa_service=qa_service,
        vector_repo=vector_repo,
        response_dispatcher=dispatcher,
    )
