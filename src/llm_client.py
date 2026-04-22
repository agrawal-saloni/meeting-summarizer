"""Gemini LLM client — single `complete()` entrypoint with retry + fallback."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config import (
    GOOGLE_API_KEY,
    LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_BASE_DELAY,
    LLM_TEMPERATURE,
)

log = logging.getLogger(__name__)

_client: genai.Client | None = None

# Status codes that indicate transient server/rate-limit issues worth retrying.
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _is_transient(exc: BaseException) -> bool:
    """True if the exception represents a retryable upstream condition."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return code in _TRANSIENT_STATUS_CODES
    return False


def _call_model(model: str, contents: str, cfg: types.GenerateContentConfig) -> str:
    """Single API call with exponential backoff + jitter on transient errors."""
    last_exc: BaseException | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = _get_client().models.generate_content(
                model=model, contents=contents, config=cfg
            )
            return response.text or ""
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient(exc) or attempt == LLM_MAX_RETRIES:
                raise
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)  # jitter
            log.warning(
                "Gemini transient error on %s (attempt %d/%d): %s — retrying in %.1fs",
                model,
                attempt,
                LLM_MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
    # Unreachable, but keeps type-checkers happy.
    raise last_exc  # type: ignore[misc]


def complete(
    system: str,
    user: str,
    json_mode: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt to Gemini and return either a raw string or parsed JSON.

    Retries transient errors (429/5xx) with exponential backoff, up to
    ``LLM_MAX_RETRIES`` attempts. If ``LLM_FALLBACK_MODEL`` is set and the
    primary model still fails with a transient error after exhausting retries,
    the fallback model is tried with the same retry policy.
    """
    primary = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "temperature": temperature,
        "max_output_tokens": LLM_MAX_TOKENS,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    cfg = types.GenerateContentConfig(**config_kwargs)

    try:
        content = _call_model(primary, user, cfg)
    except Exception as exc:
        if LLM_FALLBACK_MODEL and LLM_FALLBACK_MODEL != primary and _is_transient(exc):
            log.warning(
                "Primary model %s unavailable, falling back to %s",
                primary,
                LLM_FALLBACK_MODEL,
            )
            content = _call_model(LLM_FALLBACK_MODEL, user, cfg)
        else:
            raise

    return json.loads(content) if json_mode else content
