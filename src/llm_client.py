"""Groq LLM client — single `complete()` entrypoint with retry + fallback.

Uses Groq's hosted open-source models (Llama 3, Mixtral, Gemma, etc.) through
their OpenAI-compatible chat completions API. Free-tier friendly.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from config import (
    GROQ_API_KEY,
    LLM_FALLBACK_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_BASE_DELAY,
    LLM_TEMPERATURE,
)

log = logging.getLogger(__name__)

_client: Groq | None = None

_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _is_transient(exc: BaseException) -> bool:
    """True if the exception represents a retryable upstream condition."""
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
    return False


def _call_model(
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
) -> str:
    """Single API call with exponential backoff + jitter on transient errors."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: BaseException | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient(exc) or attempt == LLM_MAX_RETRIES:
                raise
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)  # jitter
            log.warning(
                "Groq transient error on %s (attempt %d/%d): %s — retrying in %.1fs",
                model,
                attempt,
                LLM_MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def complete(
    system: str,
    user: str,
    json_mode: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt to Groq and return either a raw string or parsed JSON.

    Retries transient errors (429/5xx) with exponential backoff, up to
    ``LLM_MAX_RETRIES`` attempts. If ``LLM_FALLBACK_MODEL`` is set and the
    primary model still fails with a transient error after exhausting retries,
    the fallback model is tried with the same retry policy.
    """
    primary = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    if json_mode:
        # Groq requires the word "json" to appear in the prompt when using
        # response_format=json_object. Most prompts already mention it; this
        # nudge keeps things robust.
        system = f"{system}\n\nRespond with a single valid JSON object."

    try:
        content = _call_model(primary, system, user, temperature, json_mode)
    except Exception as exc:
        if LLM_FALLBACK_MODEL and LLM_FALLBACK_MODEL != primary and _is_transient(exc):
            log.warning(
                "Primary model %s unavailable, falling back to %s",
                primary,
                LLM_FALLBACK_MODEL,
            )
            content = _call_model(
                LLM_FALLBACK_MODEL, system, user, temperature, json_mode
            )
        else:
            raise

    return json.loads(content) if json_mode else content
