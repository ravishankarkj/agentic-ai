import logging
import threading
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    messages: list[BaseMessage] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)


class ConversationStore:
    """In-process chat history keyed by session id, capped by turn count and idle expiry."""

    def __init__(self, max_turns: int, ttl_seconds: float) -> None:
        self._max_messages = max(max_turns, 1) * 2
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_seen = time.monotonic()
            return list(session.messages)

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._evict_expired()
            session = self._sessions.setdefault(session_id, _Session())
            session.messages.append(HumanMessage(content=question))
            session.messages.append(AIMessage(content=answer))
            if len(session.messages) > self._max_messages:
                del session.messages[: len(session.messages) - self._max_messages]
            session.last_seen = time.monotonic()
            logger.info("Conversation turn stored. session=%s messages=%s", session_id, len(session.messages))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, session in self._sessions.items() if now - session.last_seen > self._ttl_seconds]
        for key in expired:
            del self._sessions[key]
        if expired:
            logger.info("Expired idle conversation sessions. count=%s", len(expired))
