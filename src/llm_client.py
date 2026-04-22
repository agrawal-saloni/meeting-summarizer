"""Gemini LLM client — single `complete()` entrypoint."""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from config import (
    GOOGLE_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def complete(
    system: str,
    user: str,
    json_mode: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt to Gemini and return either a raw string or parsed JSON."""
    model = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "temperature": temperature,
        "max_output_tokens": LLM_MAX_TOKENS,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = _get_client().models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    content = response.text or ""
    return json.loads(content) if json_mode else content
