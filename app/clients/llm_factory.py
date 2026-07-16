import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class LLMFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _ollama_headers(self) -> dict[str, str]:
        if not self._settings.ollama_api_key:
            return {}
        return {"Authorization": f"Bearer {self._settings.ollama_api_key}"}

    def create_chat_model(self) -> BaseChatModel:
        if self._settings.llm_provider == "ollama":
            client_kwargs: dict[str, Any] = {}
            headers = self._ollama_headers()
            if headers:
                client_kwargs["headers"] = headers

            logger.info(
                "Initializing Ollama chat model. model=%s base_url=%s",
                self._settings.ollama_chat_model,
                self._settings.ollama_base_url,
            )
            return ChatOllama(
                model=self._settings.ollama_chat_model,
                base_url=self._settings.ollama_base_url,
                client_kwargs=client_kwargs or None,
                temperature=0.5,
            )

        # Lazy import keeps OpenAI provider optional at runtime.
        from langchain_openai import ChatOpenAI

        logger.info("Initializing OpenAI chat model. model=%s", self._settings.openai_chat_model)
        return ChatOpenAI(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_chat_model,
            temperature=0,
        )

    def create_embeddings(self) -> Embeddings:
        if self._settings.embedding_provider == "watsonx":
            # client_kwargs: dict[str, Any] = {}
            # headers = self._ollama_headers()
            # if headers:
            #     client_kwargs["headers"] = headers

            embed_params = {
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: self._settings.truncate_input_tokens,  # Truncate text past 512 tokens
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": False} 
            }


            logger.info(
                "Initializing Watsonx embedding model. model=%s url=%s",
                self._settings.watsonx_embedding_model,
                self._settings.watsonx_url,
            )
            return WatsonxEmbeddings(
                model_id=self._settings.watsonx_embedding_model,
                url=self._settings.watsonx_url,
                project_id=self._settings.watsonx_project_id,
                api_key=self._settings.watsonx_api_key,
                params=embed_params
            )
            # return OllamaEmbeddings(
            #     model=self._settings.ollama_embedding_model,
            #     base_url=self._settings.ollama_base_url,
            #     client_kwargs=client_kwargs or None,
            # )

        from langchain_openai import OpenAIEmbeddings

        logger.info("Initializing OpenAI embedding model. model=%s", self._settings.openai_embedding_model)
        return OpenAIEmbeddings(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_embedding_model,
        )
