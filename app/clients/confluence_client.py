import logging

from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents import Document

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class ConfluenceClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _build_loader(self, *, space_key: str | None = None, page_ids: list[str] | None = None) -> ConfluenceLoader:
        return ConfluenceLoader(
            url=self._settings.confluence_url,
            username=self._settings.confluence_username,
            api_key=self._settings.confluence_api_token,
            cloud=True,
            space_key=space_key,
            page_ids=page_ids,
            limit=self._settings.confluence_page_limit,
            include_attachments=False,
            include_comments=False,
        )

    def load_documents(self) -> list[Document]:
        if not self._settings.confluence_url:
            raise ValueError("CONFLUENCE_URL is required")
        if not self._settings.confluence_username:
            raise ValueError("CONFLUENCE_USERNAME is required")
        if not self._settings.confluence_api_token:
            raise ValueError("CONFLUENCE_API_TOKEN is required")

        logger.info("Initializing Confluence loader for url=%s", self._settings.confluence_url)

        loaded_docs: list[Document] = []

        if self._settings.confluence_space_keys_list:
            logger.info("Loading Confluence docs by space keys: %s", self._settings.confluence_space_keys_list)
        for space_key in self._settings.confluence_space_keys_list:
            loader = self._build_loader(space_key=space_key)
            loaded_docs.extend(
                loader.load()
            )

        if self._settings.confluence_page_ids_list:
            logger.info("Loading Confluence docs by page IDs. count=%s", len(self._settings.confluence_page_ids_list))
            loader = self._build_loader(page_ids=self._settings.confluence_page_ids_list)
            loaded_docs.extend(loader.load())

        if not self._settings.confluence_space_keys_list and not self._settings.confluence_page_ids_list:
            logger.info("Loading Confluence docs with default listing. limit=%s", self._settings.confluence_page_limit)
            raise ValueError("Configure CONFLUENCE_SPACE_KEYS or CONFLUENCE_PAGE_IDS for Confluence ingestion.")

        seen_doc_ids: set[str] = set()
        unique_docs: list[Document] = []
        for doc in loaded_docs:
            doc_id = str(doc.metadata.get("id") or doc.metadata.get("source") or hash(doc.page_content))
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            unique_docs.append(doc)

        logger.info("Confluence loading completed. fetched=%s unique=%s", len(loaded_docs), len(unique_docs))
        return unique_docs
