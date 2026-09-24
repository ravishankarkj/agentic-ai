import logging
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

_CONDENSE_SYSTEM = """
You rewrite follow-up messages into standalone search questions.

Given the conversation so far and the user's latest message, rewrite the latest message so it can be
understood on its own, resolving pronouns and implicit references from the conversation.
If the latest message is already standalone, return it unchanged.
Never answer the question. Return only the rewritten question, with no preamble or quotes.
""".strip()

_ANSWER_SYSTEM = """
You are an enterprise assistant answering questions from Confluence knowledge.
Use only the provided context. If context is insufficient, say you do not have enough information.
Earlier conversation is provided for reference only; the context below is the only source of facts.

Context:
{context}
""".strip()


class QAService:
    def __init__(self, retrieval_service: RetrievalService, chat_model) -> None:
        self._retrieval_service = retrieval_service
        self._chat_model = chat_model

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _ANSWER_SYSTEM),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        self._condense_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _CONDENSE_SYSTEM),
                MessagesPlaceholder("chat_history"),
                ("human", "Latest message: {question}"),
            ]
        )

    async def condense_question(self, query: str, history: list[BaseMessage] | None) -> str:
        """Rewrite a follow-up into a self-contained question so retrieval has the missing subject."""
        if not history:
            return query

        chain = self._condense_prompt | self._chat_model | StrOutputParser()
        try:
            rewritten = await chain.ainvoke({"chat_history": history, "question": query})
        except Exception:
            logger.exception("Question condensing failed. Falling back to the raw query.")
            return query

        rewritten = rewritten.strip().strip('"').strip()
        if not rewritten:
            return query

        logger.info("Condensed follow-up question. standalone=%r", rewritten)
        return rewritten

    def answer(self, query: str, history: list[BaseMessage] | None = None) -> tuple[str, list[Document]]:
        logger.info("Generating answer for query.")
        docs = self._retrieval_service.retrieve(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        chain = self._prompt | self._chat_model | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query, "chat_history": history or []})
        logger.info("Answer generation completed. context_docs=%s", len(docs))
        return answer, docs

    def retrieve_context(self, query: str) -> list[Document]:
        logger.info("Retrieving context for query.")
        docs = self._retrieval_service.retrieve(query)
        logger.info("Context retrieval completed. context_docs=%s", len(docs))
        return docs

    async def stream_answer(
        self,
        query: str,
        docs: list[Document],
        history: list[BaseMessage] | None = None,
    ) -> AsyncIterator[str]:
        logger.info("Streaming answer for query.")
        context = "\n\n".join(doc.page_content for doc in docs)
        chain = self._prompt | self._chat_model | StrOutputParser()

        async for chunk in chain.astream({"context": context, "question": query, "chat_history": history or []}):
            if chunk:
                yield chunk

        logger.info("Answer streaming completed. context_docs=%s", len(docs))
