import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from app.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CachedToken:
    value: str
    expires_at: float

    def is_valid(self, leeway_seconds: float) -> bool:
        return time.monotonic() + leeway_seconds < self.expires_at


class OAuthTokenProvider:
    """Fetches and caches an OAuth2 client-credentials bearer token for the response callback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._token: _CachedToken | None = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        leeway = self._settings.response_callback_token_refresh_leeway_seconds

        cached = self._token
        if cached is not None and cached.is_valid(leeway):
            return cached.value

        async with self._lock:
            # Another coroutine may have refreshed the token while we waited for the lock.
            cached = self._token
            if cached is not None and cached.is_valid(leeway):
                return cached.value

            token = await self._fetch_token()
            self._token = token
            return token.value

    def invalidate(self, stale_token: str) -> None:
        """Drop the cache only if it still holds the token that was rejected."""
        if self._token is not None and self._token.value == stale_token:
            self._token = None
            logger.info("Invalidated cached callback token after an authentication failure.")

    async def _fetch_token(self) -> _CachedToken:
        settings = self._settings

        if not settings.response_callback_token_url:
            raise ValueError("RESPONSE_CALLBACK_TOKEN_URL is required when callback auth mode is 'oauth'")
        if not settings.response_callback_client_id or not settings.response_callback_client_secret:
            raise ValueError(
                "RESPONSE_CALLBACK_CLIENT_ID and RESPONSE_CALLBACK_CLIENT_SECRET are required "
                "when callback auth mode is 'oauth'"
            )

        form = {
            "grant_type": "client_credentials",
            "client_id": settings.response_callback_client_id,
            "client_secret": settings.response_callback_client_secret,
        }

        logger.info("Requesting new callback bearer token. url=%s", settings.response_callback_token_url)
        async with httpx.AsyncClient(timeout=settings.response_callback_timeout_seconds) as client:
            response = await client.post(
                settings.response_callback_token_url,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.is_error:
            logger.error("Token request failed. status=%s body=%s", response.status_code, response.text[:500])
        response.raise_for_status()

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise ValueError("Token endpoint response did not contain 'access_token'")

        ttl = self._resolve_ttl(payload)
        logger.info("Fetched new callback bearer token. ttl_seconds=%s, access_token=%s", ttl, access_token)
        return _CachedToken(value=access_token, expires_at=time.monotonic() + ttl)

    def _resolve_ttl(self, payload: dict) -> float:
        expires_in = payload.get("expires_in")
        if expires_in is None:
            # Salesforce omits expires_in unless the app issues JWT-based access tokens.
            return self._settings.response_callback_token_ttl_seconds

        try:
            return float(expires_in)
        except (TypeError, ValueError):
            logger.warning("Token endpoint returned an unusable 'expires_in'. Falling back to the configured TTL.")
            return self._settings.response_callback_token_ttl_seconds
