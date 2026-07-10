from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = "Brain Box With AI"
    app_env: str = "local"
    log_level: str = "INFO"

    confluence_url: str = ""
    confluence_username: str = ""
    confluence_api_token: str = ""
    confluence_space_keys: str = ""
    confluence_page_ids: str = ""
    confluence_page_limit: int = 50

    llm_provider: Literal["ollama", "openai"] = "ollama"
    embedding_provider: Literal["ollama", "openai", "watsonx"] = "watsonx"

    ollama_base_url: str = ""
    ollama_api_key: str = ""
    ollama_chat_model: str = "gemma3"
    ollama_embedding_model: str = "nomic-embed-text"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Watsonx EMBEDDING
    watsonx_url: str
    watsonx_project_id: str
    watsonx_api_key: str
    watsonx_embedding_model: str = "ibm/granite-embedding-278m-multilingual"

    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 6
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    vector_store_dir: str = ".data/chroma"
    vector_collection_name: str = "confluence_docs"

    response_callback_url: str = "http://127.0.0.1:8000"
    response_callback_timeout_seconds: float = 10.0
    response_callback_auth_mode: Literal["none", "basic", "bearer"] = "none"
    response_callback_username: str = ""
    response_callback_password: str = ""
    response_callback_token: str = ""
    response_callback_bearer_token: str = ""

    @property
    def confluence_space_keys_list(self) -> list[str]:
        return [x.strip() for x in self.confluence_space_keys.split(",") if x.strip()]

    @property
    def confluence_page_ids_list(self) -> list[str]:
        return [x.strip() for x in self.confluence_page_ids.split(",") if x.strip()]

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 1200)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def validate_weight_range(cls, value: float) -> float:
        if value < 0:
            raise ValueError("retrieval weights cannot be negative")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
