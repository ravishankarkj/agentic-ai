import logging
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class QAService:
    def __init__(self, retrieval_service: RetrievalService, chat_model) -> None:
        self._retrieval_service = retrieval_service
        self._chat_model = chat_model

        self._prompt = ChatPromptTemplate.from_template(
            """
You are an enterprise assistant answering questions from Confluence knowledge.
Use only the provided context. If context is insufficient, say you do not have enough information.

Context:
{context}

Question:
{question}
""".strip()
        )

    def answer(self, query: str) -> tuple[str, list[Document]]:
        logger.info("Generating answer for query.")
        docs = self._retrieval_service.retrieve(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        chain = self._prompt | self._chat_model | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        logger.info("Answer generation completed. context_docs=%s", len(docs))
        return answer, docs

    def retrieve_context(self, query: str) -> list[Document]:
        logger.info("Retrieving context for query.")
        docs = self._retrieval_service.retrieve(query)
        logger.info("Context retrieval completed. context_docs=%s", len(docs))
        return docs

    async def stream_answer(self, query: str, docs: list[Document]) -> AsyncIterator[str]:
        logger.info("Streaming answer for query.")
        context = "\n\n".join(doc.page_content for doc in docs)
        chain = self._prompt | self._chat_model | StrOutputParser()

        async for chunk in chain.astream({"context": context, "question": query}):
            if chunk:
                yield chunk

        logger.info("Answer streaming completed. context_docs=%s", len(docs))
