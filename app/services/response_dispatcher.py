import base64
import logging
from typing import Any

import httpx

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class ResponseDispatcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        auth_mode = self._settings.response_callback_auth_mode.lower()

        if auth_mode == "none":
            return headers

        if auth_mode == "bearer":
            bearer_token = self._settings.response_callback_bearer_token.strip()
            if not bearer_token:
                logger.warning(
                    "Bearer callback auth is enabled but RESPONSE_CALLBACK_BEARER_TOKEN is empty. "
                    "Sending request without Authorization header."
                )
                return headers
            headers["Authorization"] = f"Bearer {bearer_token}"
            return headers

        if auth_mode != "basic":
            logger.warning("Unknown callback auth mode '%s'. Sending request without Authorization header.", auth_mode)
            return headers

        username = self._settings.response_callback_username
        password = self._settings.response_callback_password
        token = self._settings.response_callback_token
        if not username or not password:
            logger.warning(
                "Basic callback auth is enabled but username/password are missing. "
                "Sending request without Authorization header."
            )
            return headers

        password_with_token = f"{password}{token}"
        credentials = base64.b64encode(
            f"{username}:{password_with_token}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {credentials}"
        return headers

    async def _post_payload(self, payload: dict[str, Any]) -> None:
        async with httpx.AsyncClient(timeout=self._settings.response_callback_timeout_seconds) as client:
            response = await client.post(
                self._settings.response_callback_url,
                json=payload,
                headers=self._build_headers(),
            )
            logger.info(
                "Dispatched response payload to callback URL. status=%s url=%s",
                response.status_code,
                str(response.request.url),
            )
            if response.is_error:
                logger.error(
                    "Callback request failed before raise_for_status. status=%s response_body=%s",
                    response.status_code,
                    response.text,
                )
            response.raise_for_status()
            logger.info(
                "Successfully dispatched response payload to callback URL. Status code: %s, response_body=%s",
                response.status_code,
                response.text,
            )

    async def dispatch_stream_chunk(
        self,
        *,
        request_id: str,
        user_id: str,
        query: str,
        chunk: str,
        sequence: int,
    ) -> None:
        await self.dispatch(
            {
                "event": "token",
                "requestId": request_id,
                "userId": user_id,
                "response": {
                    "query": query,
                    "chunk": chunk,
                    "sequence": sequence,
                },
            }
        )

    async def dispatch_sources(
        self,
        *,
        request_id: str,
        user_id: str,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> None:
        await self.dispatch(
            {
                "event": "sources",
                "requestId": request_id,
                "userId": user_id,
                "response": {
                    "query": query,
                    "answer": answer,
                    "sources": sources,
                },
            }
        )

    async def dispatch(self, payload: dict) -> None:
        if not self._settings.response_callback_url:
            logger.info("RESPONSE_CALLBACK_URL is not configured. Skipping outbound dispatch.")
            return

        try:
            await self._post_payload(payload)
        except Exception:
            logger.exception("Failed to dispatch response payload to callback URL")
