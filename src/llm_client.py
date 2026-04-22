"""Unified LLM client — Google Gemini and Anthropic behind one `complete()` call.

Swap providers via config.LLM_PROVIDER without changing caller code.
"""

from __future__ import annotations

import json
from typing import Any

from config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


def complete(
    system: str,
    user: str,
    json_mode: bool = False,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Send a prompt and return either a raw string or parsed JSON."""
    model = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    if LLM_PROVIDER == "gemini":
        return _complete_gemini(system, user, json_mode, model, temperature)
    if LLM_PROVIDER == "anthropic":
        return _complete_anthropic(system, user, json_mode, model, temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _complete_anthropic(
    system: str, user: str, json_mode: bool, model: str, temperature: float
) -> Any:
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model,
        system=system,
        max_tokens=LLM_MAX_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    content = msg.content[0].text if msg.content else ""
    return json.loads(content) if json_mode else content


def _complete_gemini(
    system: str, user: str, json_mode: bool, model: str, temperature: float
) -> Any:
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": LLM_MAX_TOKENS,
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    client = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system,
    )
    response = client.generate_content(user)
    content = response.text if response.text else ""
    return json.loads(content) if json_mode else content
